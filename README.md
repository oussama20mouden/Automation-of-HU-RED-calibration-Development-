# Automation-of-HU-RED-calibration-Development-
Automated HU–RED calibration QA for radiotherapy CT. Python tool that extracts HU–RED calibration curves automatically from DICOM phantom scans (CIRS 062, Easy Cube, Catphan), exports CSV/JSON, generates plots, and supports monthly constancy testing vs baseline with tolerance checks.
Automated HU–RED Calibration & Monthly Constancy QA (Radiotherapy CT)

Radiotherapy delivers high doses of radiation to tumors, requiring careful optimization and quality management to maximize tumor control while minimizing normal tissue damage. To ensure dose calculation accuracy, quality assurance (QA) protocols led by certified medical physicists must keep CT-to-density calibration within tight tolerances.

This repository provides a Python-based automated quality management procedure for the monthly calibration and constancy testing of the CT Hounsfield unit to relative electron density (HU–RED) conversion. The tool implements an automated workflow to extract HU–RED calibration curves from phantom CT scans, reducing manual workload, minimizing human error, and improving reproducibility.

What the code does

From a DICOM CT series acquired with a calibration phantom, the script:
	•	Loads and sorts a CT DICOM series and converts pixel values to Hounsfield Units (HU)
	•	Automatically selects the most relevant slice (largest phantom cross-section)
	•	Segments the phantom body and detects material inserts automatically
	•	Measures mean HU and standard deviation inside circular ROIs (edge-avoiding ROI strategy)
	•	Matches measured inserts to a known RED table (configurable per phantom)
	•	Generates:
	•	CSV output (HU/STD/RED + acquisition metadata)
	•	JSON summary report
	•	Optional HU–RED curve plot (PNG)
	•	Optional ROI overlay image for QA verification
	•	Supports monthly constancy testing by comparing results to a baseline CSV and applying per-material tolerance thresholds, with pass/fail reporting and Pearson correlation.

Supported phantoms
	•	CIRS Model 062 (062M)
	•	Easy Cube
	•	Catphan (CTP404 module)

Note: Exact RED values depend on the phantom certificate and insert set. The repository supports overriding all phantom insert definitions and tolerances via an external JSON config.

Why this matters

Manual HU–RED curve generation is time-consuming and prone to user variability (ROI placement, slice selection, insert identification). Automation improves:
	•	Consistency of measurements across users and months
	•	Efficiency (analysis time reduced significantly compared to manual workflows)
	•	Traceability with standardized exports and metadata logging
	•	Safety by supporting systematic QA programs aligned with international radiotherapy recommendations

Typical workflow
	1.	Scan calibration phantom with the clinical CT protocol
	2.	Run the script on the DICOM series folder
	3.	Export curve + report
	4.	Compare to baseline for monthly constancy QA
