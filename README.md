# Automated HU–RED Calibration & Monthly Constancy QA (Radiotherapy CT)

Radiotherapy delivers high doses of radiation to tumors, requiring careful optimization and quality management to maximize tumor control while minimizing normal tissue damage. To ensure dose calculation accuracy, quality assurance (QA) protocols led by certified medical physicists must keep CT-to-density calibration within tight tolerances.

This repository provides a **Python-based automated quality management procedure** for **monthly calibration and constancy testing** of the **CT Hounsfield unit to relative electron density (HU–RED) conversion**. The tool automates the extraction of HU–RED calibration curves from phantom CT scans, reducing manual workload, minimizing human error, and improving reproducibility.

## What the code does
From a DICOM CT series acquired with a calibration phantom, the script:
- Loads and sorts a CT DICOM series and converts pixel values to **Hounsfield Units (HU)**
- Automatically selects the most relevant slice (largest phantom cross-section)
- Segments the phantom body and detects material inserts automatically
- Measures **mean HU** and **HU standard deviation** inside circular ROIs (edge-avoiding ROI strategy)
- Matches measured inserts to a **known RED table** (configurable per phantom)
- Generates:
  - **CSV** output (HU/STD/RED + acquisition metadata)
  - **JSON** summary report
  - Optional **HU–RED curve plot (PNG)**
  - Optional **ROI overlay image** for QA verification
- Supports **monthly constancy testing** by comparing results to a baseline CSV and applying per-material tolerance thresholds, with pass/fail reporting and Pearson correlation.

## Supported phantoms
- **CIRS Model 062 (062M)**
- **Easy Cube**
- **Catphan (CTP404 module)**

> Note: Exact RED values depend on the phantom certificate and insert set. The repository supports overriding phantom insert definitions and tolerances via an external JSON config.

## Why this matters
Manual HU–RED curve generation is time-consuming and prone to user variability (ROI placement, slice selection, insert identification). Automation improves:
- Consistency of measurements across users and months
- Efficiency (major reduction in analysis time compared to manual workflows)
- Traceability with standardized exports and metadata logging
- Safety by supporting systematic QA programs aligned with international radiotherapy recommendations

## Typical workflow
1. Scan the calibration phantom with the clinical CT protocol
2. Run the script on the DICOM series folder
3. Export curve + report
4. Compare to baseline for monthly constancy QA
## Paper / related publication (RPC)
A published study in *Radiation Physics and Chemistry* (RPC) provides additional context on HU–density calibration in radiotherapy and uses the **EasyCube HU–density curve as a reference** when comparing calibration curves derived from a novel phantom in pediatric radiotherapy planning.  [oai_citation:0‡ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0969806X25005602)

DOI:

https://doi.org/10.1016/j.radphyschem.2025.113068
## Example commands
```bash
# Auto-detect phantom type
python hu_red_calibration.py --dicom /path/to/dicom_series --outdir ./out --phantom auto --debug-overlay

# Force phantom type
python hu_red_calibration.py --dicom /path/to/dicom_series --outdir ./out --phantom cirs062
python hu_red_calibration.py --dicom /path/to/dicom_series --outdir ./out --phantom easycube
python hu_red_calibration.py --dicom /path/to/dicom_series --outdir ./out --phantom catphan_ctp404

# Monthly constancy test vs baseline
python hu_red_calibration.py --dicom /path/to/dicom_series --outdir ./out --phantom cirs062 --baseline ./baseline/hu_red_cirs062.csv
