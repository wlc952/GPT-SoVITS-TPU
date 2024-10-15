# GPT-SoVITS-TPU
  
## Prepare the environment

### Install dependencies

```bash
cd GPT-SoVITS-TPU
sudo chmod +x prepare.sh
./prepare.sh
```

### Download models

```bash
sudo chmod +x download.sh
./download.sh
```

## Inference on SG2300X

It takes about 20 seconds to generate a speech of about 30 words on SG2300X.

```bash
python webui_app.py
```
