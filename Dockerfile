FROM archlinux/base

RUN pacman -Syy && \
    pacman --noconfirm -S tesseract tesseract-data-eng tesseract-data-ind python python-pip opencv

COPY ./ /ktp-ocr/
COPY ./tesseract/nik /usr/share/tessdata/
RUN pip install --no-cache-dir -r /ktp-ocr/requirements.txt