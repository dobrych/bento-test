import uuid
import bentoml
import typing as t
import numpy as np
import os.path
from PIL import Image
from pathlib import Path
from typing import AsyncGenerator
from ffmpeg.asyncio import FFmpeg
from annotated_types import Le, Ge
from typing_extensions import Annotated


GRAPHIC_MODEL = "stabilityai/sdxl-turbo"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TEXT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

MAX_TOKENS = 512
CARD_DESIGN_PROMPT = """{text} beautiful cute postcard"""
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Generate sincere text for greeting card in five sentences witha a theme: {user_prompt} [/INST] """


@bentoml.service(
    traffic={"timeout": 600},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)

class SDXLTurbo:
    def __init__(self) -> None:
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            GRAPHIC_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device="cuda")
        #self.pipe.to(device="cuda")

    @bentoml.api
    async def txt2img(
            self,
            prompt: str,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 1,
            guidance_scale: float = 0.0,
    ):
        # imgs = 
        return self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).to_tuple()
        # print(imgs)
        # return imgs.images[0]


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)

class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(
            model=TEXT_MODEL,
            max_model_len=MAX_TOKENS
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)
        async for request_output in stream:
            yield request_output.outputs[0].text

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)


@bentoml.service(
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
    traffic={"timeout": 300},
)

class XTTS:
    def __init__(self) -> None:
        import torch
        from TTS.api import TTS

        self.tts = TTS(TTS_MODEL, gpu=torch.cuda.is_available())
    
    @bentoml.api
    async def get_audio(
            self,
            context: bentoml.Context,
            text: str,
            lang: str = "en",
    ) -> t.Annotated[Path, bentoml.validators.ContentType('audio/*')]:
        output_path = os.path.join(context.temp_dir, "output.wav")
        sample_path = "./female.wav"
        if not os.path.exists(sample_path):
            sample_path = "./src/female.wav"

        self.tts.tts_to_file(
            text,
            file_path=output_path,
            speaker_wav=sample_path,
            language=lang,
            split_sentences=True,
        )
        return Path(output_path)


@bentoml.service(
    traffic={"timeout": 600},
    workers=2,
    resources={"cpu": "1"}
)

class BuildGreeting:
    imggen = bentoml.depends(SDXLTurbo)
    txtgen = bentoml.depends(VLLM)
    audgen = bentoml.depends(XTTS)

    @bentoml.api
    async def generate(self, text):
        fileid = uuid.uuid4().hex
        txt = await self.txtgen.generate_text(prompt=CARD_DESIGN_PROMPT.format(text=text))
        img = await self.imggen.txt2img(prompt=txt, num_inference_steps=5, guidance_scale=5)
        aud = await self.audgen.get_audio(txt=text)

        print(txt)
        print(aud)

        videofile_path = os.path.join(
            bentoml.Context.temp_dir, 
            "{fileid}.mp4".format(fileid=fileid)
        )

        ffmpeg = (
            FFmpeg()
            .option("loop", 1)
            .option("y")
            .input(img)
            .input(aud)
            .option("c:v", "libx264")
            .option("tune", "stillimage")
            .option("c:a", "acc")
            .option("b:a", "192k")
            .option("pix_fmt", "yuv420p")
            .option("shortest")
            .output(
                videofile_path,
                # {"codec:v": "libx264"},
                # vf="scale=1280:-1",
                # preset="veryslow",
                # crf=24,
            )
        )

        mpfile = await ffmpeg.execute()
        print(videofile_path, mpfile)
        return Path(videofile_path)

