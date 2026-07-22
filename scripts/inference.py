#!/usr/bin/env python3
import argparse
from mamv_model import MAMVModel

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--document", required=True)
p.add_argument("--question", required=True)
a = p.parse_args()
print(MAMVModel.load(a.model).answer(a.document, a.question).text)
