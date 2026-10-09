from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset


try:
    import orjson
    def _loads(b: bytes):
        return orjson.loads(b)
except Exception:
    def _loads(b: bytes):
        return json.loads(b.decode("utf-8"))



def load_actions(actions_json_path: str) -> dict:
    with open(actions_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_action_matcher(api_actions: dict):
    action_names = sorted(api_actions.keys())
    patterns = [[p.lower() for p in api_actions[a]] for a in action_names]

    def match_actions(imports_dict) -> np.ndarray:
        funcs = []
        for dll in imports_dict:
            for fn in imports_dict[dll]:
                funcs.append(str(fn).lower())
        blob = "\n".join(funcs)

        out = np.zeros((len(action_names),), dtype=np.uint8)
        for i, pats in enumerate(patterns):
            for p in pats:
                if p and (p in blob):
                    out[i] = 1
                    break
        return out

    return action_names, match_actions

def check_characteristic(characteristics, char: str) -> bool:
    return any(c == char for c in characteristics)

def check_entry_point(entry: str, sections) -> bool:
    for section in sections:
        if section.get("name") == entry:
            return "MEM_EXECUTE" in section.get("props", [])
    return False


class PEMalwareOntologyTabular(Dataset):
    def __init__(
        self,
        raw_json_path: str,
        actions_json_path: str,
        normalize_numeric: bool = True,
        log1p_numeric: bool = True,
    ):
        super().__init__()
        self.raw_json_path = raw_json_path
        self.normalize_numeric = normalize_numeric
        self.log1p_numeric = log1p_numeric

        api_actions = load_actions(actions_json_path)
        self.action_names, self.match_actions = build_action_matcher(api_actions)

        self.base_cat_names = [
            "is_exe",
            "Debug",
            "Relocations",
            "Resources",
            "Signature",
            "TLS",
            "CLR",
            "NonexecutableEntryPoint",
        ]
        self.cat_feature_names = self.base_cat_names + [f"action:{a}" for a in self.action_names]

        self.num_feature_names = [
            "exports_count",
            "imports_count",
            "mz_count",
            "symbols_count",
            "path_strings_count",
            "registry_strings_count",
            "url_strings_count",
        ]

        self.num_cat = len(self.cat_feature_names)
        self.num_cont = len(self.num_feature_names)

        self._fh = None
        self._build_index_and_stats()

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __getitem__(self, idx: int):
        if self._fh is None:
            self._fh = open(self.raw_json_path, "rb")

        off = int(self.offsets[idx])
        self._fh.seek(off)
        sample = _loads(self._fh.readline())

        x_cat_np, x_num_np = self._extract_features(sample)

        if self.log1p_numeric:
            x_num_np = np.log1p(x_num_np)

        if self.normalize_numeric:
            x_num_np = (x_num_np - self.mu) / self.sigma

        x_cat = torch.from_numpy(x_cat_np.astype(np.int64))   # long for embedding
        x_num = torch.from_numpy(x_num_np.astype(np.float32))
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return x_cat, x_num, y

    def _extract_features(self, s: dict) -> tuple[np.ndarray, np.ndarray]:
        x_cat = np.zeros((self.num_cat,), dtype=np.uint8)
        x_num = np.zeros((self.num_cont,), dtype=np.float32)

        mz = int(s["strings"]["MZ"])
        paths = int(s["strings"]["paths"])
        urls = int(s["strings"]["urls"])
        reg = int(s["strings"]["registry"])

        gen = s["general"]
        exports = int(gen.get("exports", 0))
        imports = int(gen.get("imports", 0))
        symbols = int(gen.get("symbols", 0))

        
        is_dll = check_characteristic(s["header"]["coff"]["characteristics"], "DLL")
        x_cat[0] = 0 if is_dll else 1

        x_cat[1] = 1 if gen.get("has_debug", 0) == 1 else 0
        x_cat[2] = 1 if gen.get("has_relocations", 0) == 1 else 0
        x_cat[3] = 1 if gen.get("has_resources", 0) == 1 else 0
        x_cat[4] = 1 if gen.get("has_signature", 0) == 1 else 0
        x_cat[5] = 1 if gen.get("has_tls", 0) == 1 else 0

        # CLR
        clr = 0
        for d in s.get("datadirectories", []):
            if d.get("name") == "CLR_RUNTIME_HEADER" and int(d.get("size", 0)) > 0:
                clr = 1
                break
        x_cat[6] = clr

        sec = s["section"]
        entry_exec = check_entry_point(sec["entry"], sec["sections"])
        x_cat[7] = 0 if entry_exec else 1

        a = self.match_actions(s.get("imports", {}))
        x_cat[len(self.base_cat_names):] = a

        x_num[0] = float(exports)
        x_num[1] = float(imports)
        x_num[2] = float(mz)
        x_num[3] = float(symbols)
        x_num[4] = float(paths)
        x_num[5] = float(reg)
        x_num[6] = float(urls)

        return x_cat, x_num

    def _build_index_and_stats(self):
        offsets = []
        labels = []

        mean = np.zeros((self.num_cont,), dtype=np.float64)
        M2 = np.zeros((self.num_cont,), dtype=np.float64)
        n = 0

        with open(self.raw_json_path, "rb") as f:
            pos = f.tell()
            for line in f:
                try:
                    s = _loads(line)
                except Exception:
                    pos = f.tell()
                    continue

                lab = s.get("label")
                if lab not in (0, 1):
                    pos = f.tell()
                    continue

                offsets.append(pos)
                labels.append(int(lab))

                # numeric stats over the 7 numeric features
                gen = s["general"]
                mz = float(s["strings"]["MZ"])
                paths = float(s["strings"]["paths"])
                urls = float(s["strings"]["urls"])
                reg = float(s["strings"]["registry"])

                x_num = np.array([
                    float(gen.get("exports", 0)),
                    float(gen.get("imports", 0)),
                    mz,
                    float(gen.get("symbols", 0)),
                    paths,
                    reg,
                    urls,
                ], dtype=np.float64)

                if self.log1p_numeric:
                    x_num = np.log1p(x_num)

                n += 1
                delta = x_num - mean
                mean += delta / n
                delta2 = x_num - mean
                M2 += delta * delta2

                pos = f.tell()

                if n % 200_000 == 0:
                    print(f"[index] indexed {n} samples...")

        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.y = np.asarray(labels, dtype=np.uint8)

        if self.normalize_numeric and n > 1:
            var = M2 / (n - 1)
            sigma = np.sqrt(var + 1e-6)
            self.mu = mean.astype(np.float32)
            self.sigma = sigma.astype(np.float32)
        else:
            self.mu = np.zeros((self.num_cont,), dtype=np.float32)
            self.sigma = np.ones((self.num_cont,), dtype=np.float32)

        print(f"[index] done: N={len(self.offsets)}")
