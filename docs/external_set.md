# External Test Set Curation: Cleaned Non-HAM10000 ISIC 2019

## 1. Rationale & Leakage Isolation

The ISIC 2019 Challenge dataset contains 25,331 images collected from multiple clinical institutions:
- **HAM10000** (Vienna Dermatological Clinic, Austria, and Cliff Rosendahl in Queensland, Australia): 10,015 images.
- **BCN_20000** (Hospital Clínic de Barcelona, Spain): ~12,413 images.
- **MSK** (Memorial Sloan Kettering Cancer Center, USA): ~2,903 images.

Because our benchmark models are trained and cross-validated on HAM10000, evaluating directly on uncurated ISIC 2019 would cause **catastrophic patient and image leakage**. To construct a strictly leak-free external test benchmark, all HAM10000 images and patient lesions were identified and purged.

---

## 2. Filtering Protocol & Exact Counts

| Stage | Image Count | Unique Lesion IDs | Notes |
| :--- | :--- | :--- | :--- |
| **ISIC 2019 Total** | 25,331 | 11,888 | Raw competition release |
| **HAM10000 Overlap Purged** | -10,015 | -7,470 | Verified via exact `image_id` and `HAM_*` prefixes |
| **Disjoint External Test Set** | **15,316** | **4,418** | Multi-center external cohort (BCN + MSK) |

---

## 3. Class Harmonization & Label Mapping

In HAM10000, `akiec` represents actinic keratoses and intraepithelial carcinoma (including Bowen's disease / squamous cell carcinoma in situ). In ISIC 2019, Actinic Keratosis (`AK`) and Squamous Cell Carcinoma (`SCC`) were released as separate classes. Following standard clinical dermatoscopy ontology:
- `akiec` = `AK` (737) + `SCC` (431) = **1,168 images**
- `df` (Target Minority): **124 images**
- `mel` (Melanoma): **3,409 images**
- `nv` (Melanocytic Nevus): **6,170 images**
- `bcc` (Basal Cell Carcinoma): **2,809 images**
- `bkl` (Benign Keratosis): **1,525 images**
- `vasc` (Vascular Lesions): **111 images**

**Total External Evaluation Samples:** 15,316 dermoscopic images.

---

## 4. Verification & Audit

Zero images in `data/isic2019_external_metadata.csv` match any image in `data/metadata.csv`.
Zero lesion IDs in the external set overlap with any training or test fold of HAM10000.
