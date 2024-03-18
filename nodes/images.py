import os
import json
import torch
import requests
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode
from io import BytesIO
import random

class LoadImageUrl:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"url": ("STRING", { "multiline": False, })
			}
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	FUNCTION = "load_image_url"
	CATEGORY = "remote/image"
	TITLE = "Load Image (URL)"

	def load_image_url(self, url):
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
			i = Image.open(r.raw)
		image = i.convert("RGB")
		image = np.array(image).astype(np.float32) / 255.0
		image = torch.from_numpy(image)[None,]
		if 'A' in i.getbands():
			mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
			mask = 1. - torch.from_numpy(mask)
		else:
			mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
		return (image, mask)

class SaveImageUrl:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE", ),
				"url": ("STRING", { "multiline": False, }),
				"filename_prefix": ("STRING", {"default": "ComfyUI"}),
				"data_format": (["HTML_image", "Raw_data"],)
			},
			"hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
		}

	RETURN_TYPES = ()
	OUTPUT_NODE = True
	FUNCTION = "save_images"
	CATEGORY = "remote/image"
	TITLE = "Save Image (URL)"

	def save_images(self, images, url, data_format, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
		filename_prefix += "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
		results = list()
		counter = 1
		data_list = list()
		for image in images:
			data = {}
			i = 255. * image.cpu().numpy()
			img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
			meta = PngInfo()
			if prompt is not None:
				meta.add_text("prompt", json.dumps(prompt))
			if extra_pnginfo is not None:
				for x in extra_pnginfo:
					meta.add_text(x, json.dumps(extra_pnginfo[x]))
		
			file = f"{filename_prefix}_{counter:05}.png"
			buffer = BytesIO()
			img.save(buffer, "png", pnginfo=meta, compress_level=4)
			buffer.seek(0)
			encoded = b64encode(buffer.read()).decode('utf-8')
			data['file_name'] = file
			data['file_data'] = f"data:image/png;base64,{encoded}" if data_format == "HTML_image" else encoded
			counter += 1
			data_list.append(data)
			results.append({
                "filename": file,
				"type": 'upload'
            })
		with requests.post(url, json=data_list) as r:
			r.raise_for_status()
		return { "ui": { "images": results } }

class CombineImageBatch:
	"""
	This isn't needed anymore but I used it in too many places so I'm keeping it...
	"""
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images_a": ("IMAGE",),
				"images_b": ("IMAGE",),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("images",)
	FUNCTION = "combine_images"
	CATEGORY = "remote/image"
	TITLE = "Combine images"

	def combine_images(self,images_a,images_b):
		try:
			out = torch.cat((images_a,images_b), 0)
		except RuntimeError:
			print(f"Imagine size mismatch! {images_a.size()}, {images_b.size()}")
			out = images_a
		return (out,)


NODE_CLASS_MAPPINGS = {
	"LoadImageUrl" : LoadImageUrl,
	"SaveImageUrl" : SaveImageUrl,
	"CombineImageBatch" : CombineImageBatch,
}
