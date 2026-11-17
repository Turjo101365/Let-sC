"""Download the IDSSE files used by the experiments into data/external/idsse/ (gitignored).

Source: figshare doi:10.6084/m9.figshare.28196177, CC BY 4.0 (see experiments/EXTERNAL_DATA.md).
Match information and positions only (14 files, about 2.6 GB). Files already present are skipped.

    python experiments/download_idsse.py
"""

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "external", "idsse")
FILES = {
    51643472: "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml",
    51643475: "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml",
    51643478: "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml",
    51643490: "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml",
    51643487: "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml",
    51643484: "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml",
    51643481: "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml",
    51643514: "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml",
    51643517: "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml",
    51643520: "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml",
    51643523: "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml",
    51643526: "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml",
    51643529: "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml",
    51643532: "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml",
}


def main():
    os.makedirs(DEST, exist_ok=True)
    for file_id, name in FILES.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path):
            print("have", name)
            continue
        url = "https://ndownloader.figshare.com/files/{}".format(file_id)
        print("get ", name)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)


if __name__ == "__main__":
    main()
