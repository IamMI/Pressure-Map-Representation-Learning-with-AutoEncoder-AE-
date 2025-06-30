# 🧪 Pressure Map Representation Learning with AutoEncoder (AE)

## 📅 Project Info

| Creator | Date | GitLab Link |
|--------|------|-------------|
| 黄佳溢 | 2024.10.20 | [AE Project @ GitLab](https://git.nju.edu.cn/xiaomimimi/ae) |

---

## 🎯 Objective

This project explores the idea of using an **AutoEncoder (AE)** to compress pressure mat data into dense **latent variables**, and then build a separate model to **predict the latent representation** from external modalities.

> ❌ Unfortunately, the attempt did **not succeed**.

---

## 🧩 Method Summary

- Step 1: Train an AutoEncoder (AE) on pressure maps to obtain meaningful latent embeddings.
- Step 2: Build a prediction model that maps from other data (e.g., keypoints) to the AE’s latent space.
- Step 3: Use AE decoder to reconstruct pressure map from predicted latent codes.

---