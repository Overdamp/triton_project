# Triton Inference Server Integration

โปรเจคนี้ได้รวม NVIDIA Triton Inference Server เข้ามาเพื่อจัดการโมเดลต่างๆ ในระบบอ่านค่าหน้าปัดเกจอนาล็อก

## โครงสร้างโมเดลใน Triton

```
models/triton_models/
├── segmentation/
│   ├── 1/
│   │   ├── model.pt
│   │   └── config.pbtxt
│   └── config.pbtxt
├── ocr/
│   ├── 1/
│   │   ├── model.pt
│   │   └── config.pbtxt
│   └── config.pbtxt
└── superresolution/
    ├── 1/
    │   ├── model.pt  # ยังไม่ได้ใช้งาน
    │   └── config.pbtxt
    └── config.pbtxt
```

## การใช้งาน

### 1. ติดตั้ง Triton Inference Server

```bash
# ตรวจสอบ Docker
docker --version

# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
```

### 2. รัน Triton Server

```bash
# ใช้ script ที่เตรียมไว้
python triton_manager.py start --http-port 8000 --grpc-port 8001

# หรือรันด้วย Docker โดยตรง
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 3. เปิดใช้งาน Triton ใน config

แก้ไข `configs/config.yaml`:

```yaml
analog_gauge:
  use_triton: true                    # เปลี่ยนเป็น true
  triton_url: "localhost:8000"        # URL ของ Triton server
  # ... config อื่นๆ เหมือนเดิม
```

### 4. รันโปรแกรม

```bash
python main.py
```

## การจัดการโมเดล

### ตรวจสอบสถานะ Server

```bash
# ตรวจสอบว่า server ทำงานอยู่หรือไม่
python triton_manager.py status

# แสดงรายชื่อโมเดลที่โหลดแล้ว
python triton_manager.py list

# หยุด server
python triton_manager.py stop
```

### API Endpoints

- **Health Check**: `GET http://localhost:8000/v2/health/ready`
- **Model List**: `GET http://localhost:8000/v2/models`
- **Model Info**: `GET http://localhost:8000/v2/models/{model_name}`
- **Inference**: `POST http://localhost:8000/v2/models/{model_name}/infer`

## การทำงานของระบบ

1. **Local Mode** (ค่าเริ่มต้น): โหลดโมเดลโดยตรงเข้าหน่วยความจำ
2. **Triton Mode**: เชื่อมต่อกับ Triton server ผ่าน HTTP/gRPC

### ประโยชน์ของ Triton

- **Model Versioning**: จัดการเวอร์ชั่นของโมเดลได้ง่าย
- **Dynamic Batching**: รวม inference requests เป็น batch อัตโนมัติ
- **Multi-Model Serving**: เรียกใช้หลายโมเดลพร้อมกัน
- **Performance Monitoring**: ติดตามประสิทธิภาพของโมเดล
- **A/B Testing**: ทดสอบโมเดลเวอร์ชั่นใหม่กับเก่า

## การเพิ่มโมเดลใหม่

1. สร้างโฟลเดอร์ใหม่ใน `models/triton_models/`
2. วางไฟล์โมเดลและ config.pbtxt
3. รีสตาร์ท Triton server

## การแก้ปัญหา

### Triton ไม่สามารถโหลดโมเดลได้

- ตรวจสอบ config.pbtxt ให้ถูกต้อง
- ตรวจสอบ path ของโมเดล
- ดู logs ของ Triton server

### Inference ช้า

- เปิดใช้งาน GPU ใน config.pbtxt
- ปรับ dynamic batching
- เพิ่ม instance count

### Memory ไม่พอ

- ลด batch size
- ใช้ model ที่มีขนาดเล็กกว่า
- เพิ่ม RAM หรือ GPU memory

## หมายเหตุ

- Super-resolution model ยังไม่ได้ใช้งานกับ Triton อย่างเต็มรูปแบบ
- Segmentation และ OCR models ได้รับการปรับให้รองรับทั้ง Local และ Triton modes
- ระบบจะ fallback ไปใช้ Local mode อัตโนมัติถ้าเชื่อมต่อ Triton ไม่ได้