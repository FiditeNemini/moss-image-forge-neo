# 🖼️ MOSS-Image for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.1.2-brightgreen)](https://github.com/eduardoabreu81/moss-image-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate with **[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)** inside Forge Neo, using the checkpoint, preset and Generate button you already know - text-to-image with sharp text in English and Chinese, image-to-image, and layouts placed with boxes in the prompt.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (the `neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.2 - Layouts with Dynamic Prompts

- **Layouts work with sd-dynamic-prompts installed** - it takes the `{ }` out of a JSON layout before the model reads it; MOSS-Image now puts them back, so the picture is the same as without it
- **Draw your layouts** - [Regional Boxes](https://github.com/eduardoabreu81/sd-regional-boxes-neo) 0.2.0 writes them for you: draw a box per thing or text, and the layout lands in the prompt

### v0.1.1 - Works Alongside Other Extensions

- **Plays well with other extensions** - the Forge check no longer disables MOSS-Image when another extension, such as FLUX 3, loads first and hooks the loader ([#1](https://github.com/eduardoabreu81/moss-image-forge-neo/issues/1))
- **8-step LoRA** - lodestones' LoRA works with the SFT model, pictures in 8 steps instead of 50 ([how to use it](#-8-step-lora))

### v0.1.0 - First Release

- **MOSS-Image in Forge Neo** - OpenMOSS's SFT and Base models, with their Qwen3.5 9B text encoder
- **Image-to-image** and **layouts by boxes** - put things where you want them with a JSON layout in the prompt
- **moss** UI preset and **MOSS** schedule - upstream's settings, ready to generate
- **Upstream's guidance** - the negative pass skips block 22, as the official pipeline does, with a setting to change it

---

## 🎯 Features

### 🖼️ Native Generation

<p align="center"><img src=".github/images/text-rendering.jpg" alt="A refrigerator with a sticky note in Chinese, and another one with a note that reads Moss works on Forge Neo" width="720"></p>

- Runs inside txt2img and img2img - no separate tab, no extra environment
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- Uses the same autoencoder as FLUX.2 and FLUX 3 - if you have `flux2-vae`, you already have it
- Writes the text you put in quotes in the prompt, in English and in Chinese
- About a minute per 1024x1024 image at 50 steps on an A40 with the INT8 files

### 📝 Text and Posters

<p align="center"><img src=".github/images/poster-and-guidance.jpg" alt="A travel poster titled PORTO AZUL with Summer Festival 2026 at the bottom, and the refrigerator made with plain CFG" width="720"></p>

- Titles, captions and signs come out as written - put the exact words in quotes
- Any size that is a multiple of 16, such as 832x1248 for a poster

### 📐 Layout by Boxes

<p align="center"><img src=".github/images/layout.jpg" alt="A vase, a chalkboard that reads MOSS LAYOUT and a teapot placed left to right by boxes, with the Base and the SFT model" width="720"></p>

- Tell the model where each thing goes: write the whole prompt as a JSON layout, in the format of [MOSS-Image's user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#layout-prompts-experimental)
- A `summary.caption` for the whole scene, then one item per thing with its `caption`, its `text` if it is writing, and its `bbox`
- Boxes are `[top, left, bottom, right]` in pixels from the center of the picture - on a 1024x1024 image both axes go from -512 to 512
- Works with the Base and the SFT model; give boxes that make sense in the scene - things placed where they could not stand tend to follow the caption instead
- Rather draw than type? [Regional Boxes](https://github.com/eduardoabreu81/sd-regional-boxes-neo) writes the layout from boxes drawn with the mouse
- Works with sd-dynamic-prompts on: the infotext shows the layout without its `{ }`, but the model gets it whole

### 🔁 Image-to-Image

<p align="center"><img src=".github/images/img2img.jpg" alt="The refrigerator with a Moss works on Forge Neo note, and the same picture redone in img2img with a note that reads Img2img works too" width="720"></p>

- Put a picture in img2img and describe the result: with **Denoising strength** around 0.6 the scene stays and the details you change follow the prompt

### 🧭 Upstream's Guidance

- The negative-prompt pass of CFG leaves out block 22 of the model, as the official pipeline does by default
- Change it under **Settings → Stable Diffusion → [MOSS-Image] Blocks skipped in the negative-prompt pass**: other blocks, comma-separated, or empty for plain CFG
- OpenMOSS says blocks 22, 24 and 26 all help the picture, so `22, 24, 26` is worth a try
- Plain CFG looks more realistic, but some details can fall apart

### ⚡ 8-Step LoRA

- The 8-step LoRA from [lodestones](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) works with the SFT model: pictures in 8 steps instead of 50, with no negative prompt pass
- Download `lora.safetensors`, rename it to something clearer such as `moss_8step.safetensors`, put it in `models/Lora` and add it to the prompt, for example `<lora:moss_8step:1>` - or use the LoRA tab as usual
- Set **Sampler** to Euler, **Schedule type** to MOSS (or Simple), **Steps** to 8 and **CFG** to 1
- It is a work in progress: the picture is sharp and the writing is readable, but fine details and text are a little less clean than with 50 steps. With 4 steps the picture came out washed out, so stay at 8
- If you prefer `lora_latest.safetensors`, the author recommends strength 0.7

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Tells you which file is missing under **VAE / Text Encoder** instead of failing with a generic error
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/moss-image-forge-neo`
4. Click **Install** and restart the WebUI
5. Download the files from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) and place them in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| MOSS-Image checkpoint | `moss_image_sft_int8_convrot` - or `moss_image_base_int8_convrot` for the Base model | `models/Stable-diffusion` |
| Qwen3.5 9B text encoder | `qwen3.5_9b_int8_convrot` | `models/text_encoder` |
| FLUX.2 autoencoder | `flux2-vae` | `models/VAE` |

6. Restart Forge Neo, so the new text encoder shows up in the list
7. Pick the **moss** UI preset, the checkpoint, and the text encoder and autoencoder under **VAE / Text Encoder**

> [!Note]
> Tested with the INT8 files of the SFT and Base models, on an NVIDIA A40 and an RTX A6000. The BF16 files of the same repository have not been tested yet. The 8-step LoRA was tested with the INT8 SFT model on an RTX 2070 (8 GB).

---

## ⚙️ Recommended Settings

| Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- |
| Euler | MOSS | 50 | 3 |

The **moss** preset sets these values; change them on the **MOSS** page of **Settings** and the preset keeps yours. Leave the **negative prompt** empty with the SFT model; with the Base model, the user guide suggests `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`. Use sides that are a multiple of 16.

---

## 💡 Tips

- Write full captions - the scene, the objects, where they are, and the exact text in quotes
- Use Euler and the MOSS schedule; Simple gives nearly the same steps
- New text encoder files appear in the list after a restart of Forge Neo
- Before uninstalling, switch the UI preset to another one

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** by the OpenMOSS Team - the model and its reference pipeline
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous and Comfy-Org - the Qwen3.5 implementation this port follows, and the INT8 files
- **[Qwen3.5](https://huggingface.co/Qwen)** by the Qwen team - the text encoder
- **[lodestones](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)** - the 8-step LoRA
- **[Transformers](https://github.com/huggingface/transformers)** by Hugging Face - the Qwen3.5 reference

The MOSS-Image weights are released by the OpenMOSS Team under the Apache License 2.0; this extension does not include them.

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/moss-image-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/moss-image-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
