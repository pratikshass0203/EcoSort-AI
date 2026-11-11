# EcoSort AI — Dataset Guide & Taxonomy Mapping

**Bharat AI-SoC Challenge 2026–2027**  
**Problem Statement 2:** Smart Recycling and Waste-Segregation Assistant

---

## 1. Overview & Waste Categorization Standards

EcoSort AI addresses solid waste segregation at the source by categorizing waste into **four primary municipal and regulatory categories**:

1. **Biodegradable (Organic)**
2. **Recyclable (Dry Recyclables)**
3. **Electronic Waste (E-Waste)**
4. **Hazardous Waste (Domestic Hazardous)**

To train a computer vision classifier capable of identifying items across these categories, suitable publicly available datasets must be mapped into this 4-class taxonomy.

---

## 2. Public Dataset Evaluation & Licensing

### A. Evaluated Datasets

| Dataset | Primary Focus | Image Count | License | Suitability Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **TrashNet** (Stanford / Gary Thung & Mingke Wang) | Recyclable waste sorting on white backgrounds | 2,527 images | Open / MIT-like | **High** for dry recyclables (glass, paper, cardboard, plastic, metal).<br>❌ *Does not contain organic, e-waste, or hazardous classes.* |
| **Garbage Classification Dataset** (Kaggle / C. Yang) | Municipal waste across 12 categories | ~15,150 images | Open Dataset (CC0 / Public Domain) | **High** for biodegradable (biological) and recyclable sub-types. Contains battery class.<br>⚠️ *Contains out-of-distribution classes (clothes, shoes).* |
| **TACO** (Trash Annotations in Context / Pedro F. Proença & Pedro Simões) | Litter in natural environments (indoor/outdoor) | ~1,500 images | MIT License | **High** for contextual variations and hazard items (aerosol cans, chemical containers). Requires bounding box/crop extraction. |
| **Drinking Waste Dataset** | Beverage containers | ~4,800 images | Open / Academic | **Medium**; provides good augmentation for PET bottles and aluminum cans. |

---

## 3. Class Mapping to EcoSort 4-Category Taxonomy

Below is the formal mapping dictionary used to aggregate public datasets into EcoSort's four classes:

### Category 0: `biodegradable` (Green Bin)
- **Included Sub-classes:**
  - Food scraps, fruit peels, vegetable waste
  - Bread, grains, cooked food leftovers
  - Garden leaves, grass cuttings, twigs, flowers
  - Coffee grounds, tea leaves, eggshells
- **Source Mappings:**
  - Kaggle Garbage Classification: `biological`
  - TACO: `Food waste`, `Organic waste`

### Category 1: `recyclable` (Blue Bin)
- **Included Sub-classes:**
  - Plastic bottles (PET / HDPE), plastic jars, clean plastic food trays
  - Clean paper, newspapers, magazines, office paper
  - Corrugated cardboard, cereal boxes, paperboard
  - Aluminum cans, tin cans, aerosol cans (empty), metal bottle caps
  - Glass beverage bottles, food jars (clear, amber, green)
- **Source Mappings:**
  - TrashNet: `glass`, `paper`, `cardboard`, `plastic`, `metal`
  - Kaggle: `cardboard`, `glass`, `brown-glass`, `green-glass`, `white-glass`, `metal`, `paper`, `plastic`
  - TACO: `Aluminium foil`, `Clear plastic bottle`, `Other plastic bottle`, `Plastic bottle cap`, `Drink can`, `Food can`, `Glass bottle`, `Broken glass`, `Carded pack`

### Category 2: `electronic` (Yellow / E-Waste Bin)
- **Included Sub-classes:**
  - Discarded smartphones, feature phones, tablets
  - Printed circuit boards (PCBs), RAM sticks, computer hardware
  - USB cables, power adapters, charging cords, earphones
  - Small consumer batteries (AA, AAA, button cells, lithium-ion phone batteries)
  - Remote controls, electronic toys, smart cards
- **Source Mappings:**
  - Kaggle: `battery`
  - TACO: `Battery`
  - Custom / Curated Edge Samples: Mobile phone parts, circuit boards, USB connectors

### Category 3: `hazardous` (Red / Black Bin)
- **Included Sub-classes:**
  - Chemical, solvent, acid, and pesticide containers
  - Fluorescent light tubes, CFL bulbs, mercury thermometers
  - Expired pharmaceuticals, medicine blister packs, syringes
  - Paint cans, varnish containers, lacquer thinners
  - Motor oil and vehicle lubricant bottles
- **Source Mappings:**
  - TACO: `Aerosol` (chemical), `Blister pack` (medical)
  - Custom / Curated Edge Samples: Chemical bottles, fluorescent tubes, paint cans

---

## 4. Unmapped & Out-of-Distribution (OOD) Classes

Certain objects commonly found in municipal waste streams **cannot be reliably mapped** into the four primary categories or would introduce severe label ambiguity:

| Class | Source Dataset | Reason for Exclusion / Handling |
| :--- | :--- | :--- |
| **`clothes` / Textiles** | Kaggle | Textiles require specialized municipal textile recycling and do not decompose in ordinary compost nor melt with PET plastics. |
| **`shoes` / Footwear** | Kaggle | Composite multi-material items (rubber, synthetic leather, foam, metal eyelets) that cannot be sorted into single-stream bins. |
| **`trash` / General Dirt** | TrashNet, Kaggle | Heavily mixed or unidentifiable debris. Used exclusively for **uncertainty gate evaluation** to confirm the model triggers the `UNCERTAIN` state. |
| **Sanitary Waste** | Municipal streams | Biohazardous personal hygiene items require incineration; excluded from general consumer vision sorting to avoid bio-handling hazards. |

---

## 5. Dataset Directory Structure & Splitting Protocol

The repository implements a strict, leak-free stratified data split protocol:

```
data/
├── raw/                               # Unprocessed downloaded source archives (git-ignored)
├── processed/                         # Standardized 224x224 RGB datasets
│   ├── train/ (70%)                   # Model parameter optimization
│   │   ├── biodegradable/
│   │   ├── recyclable/
│   │   ├── electronic/
│   │   └── hazardous/
│   ├── val/ (15%)                     # Hyperparameter tuning & early stopping
│   │   ├── biodegradable/
│   │   ├── recyclable/
│   │   ├── electronic/
│   │   └── hazardous/
│   └── test/ (15%)                    # Final generalization evaluation
│       ├── biodegradable/
│       ├── recyclable/
│       ├── electronic/
│       └── hazardous/
└── samples/                           # Curated sample validation images for tests & demos
    ├── sample_biodegradable.jpg
    ├── sample_recyclable.jpg
    ├── sample_electronic.jpg
    └── sample_hazardous.jpg
```

### Preprocessing & Normalization Standards
- **Resolution:** All images are resized to $224 \times 224$ pixels using bicubic/bilinear interpolation while maintaining aspect ratio via center cropping or padded resizing.
- **Color Format:** 3-channel RGB (grayscale images converted to RGB).
- **Normalization:** ImageNet distribution statistics:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 6. Synthetic Dataset Protocol for Automated Testing

To ensure the automated test suite (`tests/test_vision_pipeline.py`) can run **100% offline without downloading multi-gigabyte external archives**, `src/vision/dataset.py` includes a deterministic synthetic dataset generator:
- Generates reproducible, colored $224 \times 224$ synthetic test images for each of the 4 classes.
- Used to test data loaders, batching, forward passes, loss calculation, backpropagation, and early stopping in CI and testing workflows.

