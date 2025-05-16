import cv2
import streamlit as st
import zipfile
import os
import shutil
from pathlib import Path

from src.config import settings
from src.schemas import SegmentorConfig
from src.segmentor import AutomaticSegmentor
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


if __name__ == "__main__":
    st.title("Détourage automatique")
    uploaded_file = st.file_uploader("Toutes les photos à détourer, dans un dossier compressé", type="zip")
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = None

    if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded:
        progress_bar = st.progress(0, text="Traitement en cours...")
        st.session_state.last_uploaded = uploaded_file
        target_dir = Path("/tmp/seg/")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "result").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(uploaded_file, 'r') as zip:
            zip.extractall(target_dir)
        with open(settings.config_file_path, "r") as f:
            config = SegmentorConfig.model_validate_json(f.read())
        seg = AutomaticSegmentor(config)
        seg.load()
        img_paths = list(target_dir.glob("*.jpg"))
        total = len(img_paths)

        for i, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            masked_img = seg.infer(img, Path(f"{target_dir}/result/"))
            cv2.imwrite(Path(f"{target_dir}/result/{img_path.name}"), masked_img)
            progress_bar.progress((i + 1) / total, text=f"Traitement de l'image {i + 1}/{total}")
        progress_bar.empty()

        shutil.make_archive(target_dir / "result", "zip", target_dir / "result")

        with open(target_dir / "result.zip", "rb") as f:
            st.session_state.zip_bytes = f.read()
        st.success("✅ Traitement terminé.")
        
    if st.session_state.zip_bytes:
        st.download_button(
            label="Télécharger le ZIP",
            data=st.session_state.zip_bytes,
            file_name="data.zip",
            mime="application/zip"
        )