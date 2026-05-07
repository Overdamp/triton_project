# สรุปโปรเจค Gauge

## ภาพรวม
โปรเจคนี้เป็นระบบอ่านค่าหน้าปัดเกจอนาล็อก (Analog Gauge) ด้วยภาพถ่าย โดยใช้โมเดล segmentation, OCR และการคำนวณมุม/ค่าเพื่อหาค่าที่ถูกต้องจากเข็ม

ตัวโปรเจคถูกออกแบบให้ทำงานแบบ batch inference จากโฟลเดอร์รูปภาพหนึ่งไปยังอีกโฟลเดอร์หนึ่ง พร้อมเก็บภาพ debug report เมื่อเปิดโหมด debug

---

## ไฟล์หลัก

### `main.py`
- เป็น entry point ของโปรเจค
- โหลดค่า configuration จาก `configs/config.yaml`
- สร้าง logger
- สร้าง `AnalogGaugeTask` เพื่อโหลดโมเดลและเตรียม pipeline
- ใช้โฟลเดอร์ input/output ที่กำหนดไว้ตรงโค้ด (ฮาร์ดโค้ด)
- อ่านรูปภาพทั้งหมดจากโฟลเดอร์ input
- วนประมวลผลภาพทีละไฟล์ด้วย `AnalogGaugeTask.execute()`
- บันทึกผลลัพธ์และ debug image ไปยังโฟลเดอร์ output

### `configs/config.yaml`
- กำหนดค่า logging และค่าโมเดลสำหรับโพรเซส
- ตัวอย่างค่าที่สำคัญ:
  - `analog_gauge.device`: device สำหรับ OCR
  - `analog_gauge.ocr_model_dir`: โฟลเดอร์ที่เก็บโมเดล OCR
  - `analog_gauge.segmentation.model_path`: พาธไฟล์โมเดล segmentation
  - `analog_gauge.segmentation.conf`, `iou`, `verbose`
  - `analog_gauge.superresolution.model_name`: โมเดล SR สำหรับช่วย OCR
  - `analog_gauge.debug.enabled`: เปิด/ปิดการสร้าง debug report
  - `analog_gauge.debug.output_dir`: โฟลเดอร์เก็บไฟล์ debug

---

## โฟลเดอร์สำคัญ

### `tasks/analog_gauge_task.py`
- เป็น pipeline หลักของการอ่าน gauge
- สร้าง instance ของ:
  - `GaugeSegmentor` สำหรับ segmentation
  - `EllipseFitter` สำหรับหา ellipse ของส่วนต่างๆ
  - `GaugeCalculator` สำหรับคำนวณค่า gauge จากมุมเข็มและค่าที่ OCR อ่านได้
  - `DoctrOCR` สำหรับอ่านตัวเลข/หน่วยจากภาพ
  - `GaugeScaleSuperResolution` สำหรับปรับภาพก่อน OCR (ถ้าใช้งาน)
  - `GaugeDebugger` สำหรับสร้างภาพ debug report
- ลำดับงานหลัก:
  1. segmentation
  2. ellipse fitting
  3. OCR
  4. gauge reading / calibration
  5. บันทึก debug report

### `libs/analog_gauge`
ประกอบด้วยโมดูลหลักสำหรับ pipeline:
- `segmentation.py`: โหลด YOLO/segment model และแปลงผล mask เป็น polygon
- `ellipsefit.py`: ฟิต ellipse จากจุด mask ด้วย `skimage.measure.EllipseModel`
- `gauge_cal.py`: คำนวณมุมเข็ม, แปลงมุมให้ถูกต้องตาม perspective, รัน RANSAC/linear fit เพื่อหา value
- `ocr_ai.py`: โหลดโมเดล Doctr OCR และทำนายค่าตัวเลขจากภาพ
- `gauge_debug.py`: สร้างภาพ debug report แบบ panel ที่แสดงหลายขั้นตอนของ pipeline
- `superresolution.py`: สนับสนุนโมเดล super-resolution เพื่อช่วยเพิ่มความชัดก่อน OCR (ถูกเรียกใช้งานในโค้ด)
- `detection.py`: โมดูลตรวจจับ object ด้วย YOLO แต่ใน pipeline หลักตอนนี้ไม่เห็นเรียกใช้งานโดยตรง
- `visualizer.py`: มี utility สำหรับแสดงผลและช่วยวาดภาพ แต่ไม่ได้ถูกอ่านทั้งหมดในสรุปนี้

---

## กระบวนการทำงานหลัก

1. โหลด configuration และ logger
2. สร้าง `AnalogGaugeTask` แล้วโหลด:
   - segmentation model
   - OCR model
   - super-resolution model (ถ้ามี)
3. สำหรับทุกภาพในโฟลเดอร์ input:
   - อ่านภาพเข้าเป็น `numpy.ndarray`
   - เรียก `AnalogGaugeTask.execute()`
     - ทำ segmentation เพื่อค้นหาชิ้นส่วนเช่น needle, scale number, max/min, unit
     - fit ellipse จาก segmentation mask
     - ทำ OCR บนองค์ประกอบที่เกี่ยวข้อง
     - คำนวณค่าจากตำแหน่งเข็มและค่าที่อ่านได้
     - สร้างรายงาน debug
   - บันทึกผลลัพธ์และภาพ debug

---

## จุดสังเกต

- `main.py` ใช้ path input/output แบบฮาร์ดโค้ด หากต้องการใช้งานทั่วไปควรเปลี่ยนเป็น parameter หรือ config
- ระบบสามารถเปิดโหมด debug เพื่อสร้างภาพรายงานแบบละเอียด
- pipeline ออกแบบเป็น modular: `AnalogGaugeTask` เชื่อมหลายโมดูลย่อยเข้าด้วยกัน
- มีโมดูลที่ใช้ ultralytics YOLO ทั้ง segmentation และ detection
- OCR ใช้ Doctr และโหลด checkpoint จาก `models/analog_gauge_model/ocr`

---

## สรุป
โปรเจคนี้ทำหน้าที่อ่านค่าจากหน้าปัดเกจอนาล็อก โดยใช้:
- segmentation เพื่อแยกเข็มและตัวเลข
- ellipse fitting เพื่อหาแกนหลักของหน้าปัด
- OCR เพื่ออ่านค่าตัวเลขและหน่วย
- การคำนวณมุมและการปรับ perspective เพื่อหาค่า gauge ที่ถูกต้อง

ไฟล์สำคัญที่สุดคือ `main.py`, `configs/config.yaml`, `tasks/analog_gauge_task.py` และโฟลเดอร์ `libs/analog_gauge`
