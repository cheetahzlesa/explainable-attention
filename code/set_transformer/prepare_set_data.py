from __future__ import annotations

from collections.abc import Iterator
from collections import defaultdict
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import urllib
import random

try:
    import orjson
    def _loads(b: bytes):
        return orjson.loads(b)
except Exception:
    def _loads(b: bytes):
        return json.loads(b.decode("utf-8"))
    
section_names = json.load(open("section_names.json", "r", encoding="utf-8"))["names"]
entropy_threshold = 7.0

    
def check_individual(individual, section_name = False):
    individual = individual.replace("<", "")
    individual = individual.replace(">", "")
    individual = individual.lower()

    if section_name:
        individual = individual.replace(".", "")

    return urllib.parse.quote(individual)

def check_section_property(section, p):
    for prop in section["props"]:
       if prop == p:
           return True
    
    return False

def check_section_entropy(entropy):
    if entropy >= entropy_threshold:
        return True
    else:
        return False

def check_section_name(section):
    section_name = section["name"].lower()

    for sn in section_names:
        if section_name.find(sn) != -1:
            return True
    
    return False

def check_section_wx(section):
    write = False
    execute = False

    for prop in section["props"]:
        if prop == "MEM_WRITE":
            write = True
        if prop == "MEM_EXECUTE":
            execute = True
    
    if write and execute:
        return True
    else:
        return False



class PEMalwareOntologySet(Dataset):
    def __init__(
        self,
        raw_json_path: str,
        binar_entropy: bool = False
    ):    
        super().__init__()
        self.raw_json_path = raw_json_path
        self.binar_entropy = binar_entropy

        self.lenghts = []


        self.num_features = 9

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

        if sample["section"]["sections"] == []:
            x_features = np.zeros((1, self.num_features,), dtype=np.float32)
        else:
            x_features = self._extract_sections(sample)

            if self.binar_entropy == False:
                x_features[: ,-1] = np.log1p(x_features[:, -1])

                x_features[: , -1] = (x_features[:, -1] - self.mu) / self.sigma


        x_features = torch.from_numpy(x_features.astype(np.float32)) # all are binary values, only last one is float 
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return x_features, y



    def _extract_sections(self, sample: dict) ->np.ndarray:
        x_features = []
     
        # 4x section flag, 3x one-hot section type, 1x section name, 1x entropy (tento jediny feature by mal byt spojity)
        for s in sample["section"]["sections"]:        
            pe_hash = sample["md5"]
            s_name = check_individual(s["name"], True)
            s_entropy = float(s["entropy"])
            section = np.zeros((self.num_features,), dtype=np.float32)
            

            if check_section_name(s):
                section[0] = 1
            
            if check_section_property(s, "MEM_EXECUTE"):
                section[1] = 1
            
            if check_section_property(s, "MEM_WRITE"):
                section[2] = 1
            
            if check_section_property(s, "MEM_READ"):
                section[3] = 1
            
            if check_section_property(s, "MEM_SHARED"):
                section[4] = 1
            

            if check_section_property(s, "CNT_CODE"):
                section[5] = 1
            
            if check_section_property(s, "CNT_INITIALIZED_DATA"):
                section[6] = 1
            
            if check_section_property(s, "CNT_UNINITIALIZED_DATA"):
                section[7] = 1
            
            if self.binar_entropy:
                section[8] = int(check_section_entropy(s_entropy))
            else:
                section[8] = s_entropy
            
            x_features.append(section)
            
            

        return np.array(x_features, dtype=np.float32)
    
    def _build_index_and_stats(self):
        offsets = []
        labels = []
        mean = 0
        M2 = 0
        n = 0
        index = 0


        with open(self.raw_json_path, "rb") as f:
            pos = f.tell()
            for line in f:
                try:
                    s = _loads(line)
                except Exception:
                    pos = f.tell()
                    continue

                lab = s.get("label")
                

                offsets.append(pos)
                labels.append(int(lab))

                
                s_section_len = max(1, len(s["section"]["sections"]))

                self.lenghts.append(s_section_len)

                if self.binar_entropy == False:
                    x_num = np.array([float(sample["entropy"]) for sample in s["section"]["sections"]], dtype=np.float64)
                    x_num = np.log1p(x_num)
                    for ent in x_num:
                        n += 1
                        delta = ent - mean
                        mean += delta / n
                        delta2 = ent - mean
                        M2 += delta * delta2

                pos = f.tell()

                if index % 200_000 == 0:
                    print(f"[index] indexed {index} samples...")
                
                index +=1

        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.y = np.asarray(labels, dtype=np.uint8)

        if self.binar_entropy == False and n > 1:
            var = M2 / (n - 1)
            sigma = np.sqrt(var + 1e-6)
            self.mu = mean
            self.sigma = sigma.astype(np.float32)
        else:
            self.mu = 0
            self.sigma = 0

        print(f"[index] done: N={len(self.offsets)}")


class BucketBatchSampler(Sampler):
    def __init__(self, buckets : list, batch_size, shuffle=True):
        super().__init__()

        self.buckets = defaultdict(list)

        for i, val in enumerate(buckets):
            self.buckets[val].append(i)

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        all_batches = []

        for no_sections, indexs in self.buckets.items():
            indexs = indexs.copy()

            if self.shuffle:
                random.shuffle(indexs)
            
            for start_index in range(0, len(indexs), self.batch_size):
                all_batches.append(indexs[start_index : start_index + self.batch_size])

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        total_batches = 0

        for indexs in self.buckets.values():
            total_batches += (len(indexs) + self.batch_size-1) // self.batch_size

        return total_batches