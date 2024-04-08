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
You are a helpful, respectful and honest assistant that gives straight answers without explanation. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

What is a good one sentence greeting card text for: {user_prompt}. Provide just the text of the card without explanation. Do not read back the prompt. [/INST] """


@bentoml.service(
    traffic={"timeout": 10000},
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
    def txt2img(
            self,
            prompt: str,
            num_inference_steps: Annotated[int, Ge(1), Le(15)] = 1,
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
        "timeout": 10000,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)

class VLLM:
    def __init__(self) -> None:
        from vllm import EngineArgs, LLMEngine

        ENGINE_ARGS = EngineArgs(
            dtype='float16',
            gpu_memory_utilization=0.5,
            max_parallel_loading_workers=1,
            model=TEXT_MODEL,
            max_model_len=MAX_TOKENS
        )

        self.engine = LLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    def generate_text(
        self,
        prompt: str = "Hello",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> str:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)
        while True:
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    return request_output.outputs[0].text
            if not (self.engine.has_unfinished_requests()):
                break

@bentoml.service(
    resources={
        "gpu": 1,
    },
    traffic={"timeout": 10000},
)

class XTTS:
    def __init__(self) -> None:
        import torch
        from TTS.api import TTS

        self.tts = TTS(TTS_MODEL, gpu=torch.cuda.is_available())

    @bentoml.api
    def get_audio(
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
        print(output_path)
        return Path(output_path)


@bentoml.service(
    traffic={"timeout": 10000},
    workers=1,
    resources={"cpu": "1"}
)

class BuildGreeting:
    imggen = bentoml.depends(SDXLTurbo)
    txtgen = bentoml.depends(VLLM)
    audgen = bentoml.depends(XTTS)

    @bentoml.api
    async def generate(self, context: bentoml.Context, text):
        print('text: ', text, type(text))
        fileid = uuid.uuid4().hex
        txt = self.txtgen.generate_text(prompt=CARD_DESIGN_PROMPT.format(text=text))
        print(txt)
        aud = self.audgen.get_audio(text=txt)
        img = self.imggen.txt2img(prompt=f"Beautiful birtday postcard: {text}", num_inference_steps=15, guidance_scale=5.0)

        print(aud)
        print(img)

        # save image to temp file
        img = img[0][0]
        img_path = os.path.join(
            Path(context.temp_dir),
            Path("{fileid}.png".format(fileid=fileid))
        )
        img.save(img_path)

        videofile_path = os.path.join(
            Path(context.temp_dir),
            Path("{fileid}.mp4".format(fileid=fileid))
        )

        ffmpeg = (
            FFmpeg()
            .option("y")
            .option("loop", 1)
            .input(img_path)
            .input(aud)
            .output(videofile_path,
                {"c:v": "libx264", "tune": "stillimage", "c:a": "aac", "b:a": "192k", "pix_fmt": "yuv420p", "shortest": None}
            )
        )

        mpfile = await ffmpeg.execute()
        print(videofile_path, mpfile)
        return Path(videofile_path)
