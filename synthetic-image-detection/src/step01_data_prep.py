"""
Step 01 — 이미 정리된 ZIP 2개를 풀고 최종 dataset.csv 생성

입력 ZIP 구조:
  defacto_15000_probe.zip
  └─ defacto_15000_probe/
     ├─ images/composite/...
     ├─ masks/probe_mask/...
     └─ defacto_15000_probe_dataset_info.csv

  coco_real_15000.zip
  └─ coco_real_15000/
     ├─ images/real/...
     ├─ masks/real/zero_mask_256.png
     └─ coco_real_15000_dataset_info.csv

출력:
  /content/drive/MyDrive/forensic_project/data/dataset.csv
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/forensic_project")
DEFAULT_DEFACTO_ZIP = "/content/drive/MyDrive/forensic_uploads/defacto_15000_probe.zip"
DEFAULT_COCO_ZIP = "/content/drive/MyDrive/forensic_uploads/coco_real_15000.zip"
DEFAULT_EXTRACT_DIR = "/content/datasets"

REQUIRED_COLS = [
    "image_path", "mask_path", "label", "split",
    "dataset_name", "manipulation_type", "group_id",
]


def unzip_clean(zip_path: str, extract_dir: Path) -> None:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP 파일 없음: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"압축 해제: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def find_single_csv(root: Path, expected_name: str) -> Path:
    candidates = list(root.rglob(expected_name))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{expected_name} 파일을 정확히 1개 찾지 못했습니다. 찾은 개수={len(candidates)}, root={root}"
        )
    return candidates[0]


def dataset_root_from_csv(csv_path: Path) -> Path:
    return csv_path.parent


def fix_path(root: Path, p: str, kind: str) -> str:
    p = str(p).replace("\\", "/")
    # Windows 절대 경로 또는 Colab 절대 경로가 들어온 경우도 처리
    if p.startswith("/content/"):
        return p
    if len(p) > 1 and p[1] == ":":
        # 로컬 Windows 절대 경로는 Colab에서 쓸 수 없으므로 zip 내부 구조 기준으로 복구한다.
        name = Path(p).name
        if kind == "image":
            preferred_roots = [root / "images"]
        else:
            preferred_roots = [root / "masks"]
        for rr in preferred_roots:
            matches = list(rr.rglob(name)) if rr.exists() else []
            if len(matches) == 1:
                return str(matches[0])
        matches = list(root.rglob(name))
        if len(matches) == 1:
            return str(matches[0])
        raise FileNotFoundError(f"Windows 경로를 Colab 경로로 복구 실패: {p}, kind={kind}, matches={len(matches)}")
    return str(root / p)


def load_and_fix(csv_path: Path, label: int) -> pd.DataFrame:
    root = dataset_root_from_csv(csv_path)
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path.name}에 필요한 컬럼이 없습니다: {missing_cols}")

    df = df[REQUIRED_COLS].copy()
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) != {label}:
        raise ValueError(f"{csv_path.name} label이 예상과 다릅니다. expected={label}, actual={df['label'].unique()}")

    df["image_path"] = df["image_path"].apply(lambda x: fix_path(root, x, "image"))
    df["mask_path"] = df["mask_path"].apply(lambda x: fix_path(root, x, "mask"))
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    print("\n=== dataset 검증 ===")
    print("전체 개수:", len(df))
    print(pd.crosstab(df["split"], df["label"]))

    expected = {
        ("train", 0): 10500, ("train", 1): 10500,
        ("validation", 0): 3000, ("validation", 1): 3000,
        ("test", 0): 1500, ("test", 1): 1500,
    }
    ct = pd.crosstab(df["split"], df["label"])
    for (split, label), n in expected.items():
        actual = int(ct.loc[split, label]) if split in ct.index and label in ct.columns else 0
        if actual != n:
            raise ValueError(f"split/label 개수 오류: {split}, label={label}, expected={n}, actual={actual}")

    img_exists = df["image_path"].apply(os.path.exists)
    mask_exists = df["mask_path"].apply(os.path.exists)
    print("이미지 존재율:", img_exists.mean())
    print("마스크 존재율:", mask_exists.mean())
    if not img_exists.all():
        print(df.loc[~img_exists, "image_path"].head(10).to_string(index=False))
        raise FileNotFoundError("존재하지 않는 이미지 경로가 있습니다.")
    if not mask_exists.all():
        print(df.loc[~mask_exists, "mask_path"].head(10).to_string(index=False))
        raise FileNotFoundError("존재하지 않는 마스크 경로가 있습니다.")

    leak = df.groupby("group_id")["split"].nunique()
    leak = leak[leak > 1]
    print("여러 split에 섞인 group_id 수:", len(leak))
    if len(leak) > 0:
        print(leak.head(20).to_string())
        raise ValueError("group_id leakage가 있습니다. split을 다시 만들어야 합니다.")

    dup = df["image_path"].duplicated().sum()
    print("중복 image_path 수:", int(dup))
    if dup:
        raise ValueError("중복 이미지가 있습니다.")

    # 참고용: COCO real ID와 DEFACTO source ID가 겹치는지 확인한다.
    # 계획서의 prefix 설계상 C_/D_는 별도 group으로 처리하지만, 발표 때 데이터 누수 질문이 나오면 이 값을 언급하면 된다.
    coco_ids = set(df.loc[df["label"] == 0, "group_id"].astype(str).str.replace("^C_", "", regex=True))
    defacto_ids = set(df.loc[df["label"] == 1, "group_id"].astype(str).str.replace("^D_", "", regex=True))
    overlap = coco_ids & defacto_ids
    print("COCO real ID와 DEFACTO source ID 겹침 수(참고):", len(overlap))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defacto_zip", default=DEFAULT_DEFACTO_ZIP)
    parser.add_argument("--coco_zip", default=DEFAULT_COCO_ZIP)
    parser.add_argument("--extract_dir", default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--clean", action="store_true", help="기존 /content/datasets 폴더를 삭제 후 다시 풉니다.")
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    if args.clean and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    unzip_clean(args.defacto_zip, extract_dir)
    unzip_clean(args.coco_zip, extract_dir)

    defacto_csv = find_single_csv(extract_dir, "defacto_15000_probe_dataset_info.csv")
    coco_csv = find_single_csv(extract_dir, "coco_real_15000_dataset_info.csv")
    print("\nDEFACTO CSV:", defacto_csv)
    print("COCO CSV:", coco_csv)

    defacto = load_and_fix(defacto_csv, label=1)
    coco = load_and_fix(coco_csv, label=0)

    df = pd.concat([coco, defacto], ignore_index=True)
    # split 순서와 label 균형 확인용으로 정렬하지 않고 섞어서 저장
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    validate_dataset(df)

    out_path = BASE_DIR / "data" / "dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n✅ 최종 dataset.csv 저장 완료:", out_path)


if __name__ == "__main__":
    main()
