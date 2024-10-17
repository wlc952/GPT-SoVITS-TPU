# GPT-SoVITS-TPU

![image](https://github.com/user-attachments/assets/42008f4f-ad62-4561-ae67-2d24a2516e81)
  
## Prepare the environment

### Install dependencies

```bash
git clone https://github.com/wlc952/GPT-SoVITS-TPU.git
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

```bash
python webui_app.py
```

It takes about 20 seconds to generate a speech of about 35 words on SG2300X.

![image](https://github.com/user-attachments/assets/00f28642-2f0a-4640-ab9d-b0cc254f0a4b)
