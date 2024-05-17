import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import io
import gc

import deep_danbooru_model

app = Flask(__name__)

model = deep_danbooru_model.DeepDanbooruModel()

# Check the mode from environment variable
mode = os.getenv('MODE', 'normal')

if mode == 'lowvram':
  model.load_state_dict(torch.load('model-resnet_custom_v3.pt', map_location='cpu'))
else:
  model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))
  model.cuda()

model.eval()
model.half()

def process_image(image):
  if mode == 'lowvram':
    model.cuda()

  pic = Image.open(image).convert("RGB").resize((512, 512))
  a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

  with torch.no_grad(), torch.autocast("cuda"):
    x = torch.from_numpy(a).cuda()

    y = model(x)[0].detach().cpu().numpy()

  results = {}
  for i, p in enumerate(y):
    if p >= 0.5:
      results[model.tags[i]] = float(p)

  if mode == 'lowvram':
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

  return results

@app.route('/predict', methods=['POST'])
def predict():
  if 'file' not in request.files:
    return jsonify({'error': 'no file'}), 400

  file = request.files['file']
  img = io.BytesIO(file.read())
  results = process_image(img)

  return jsonify(results)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
