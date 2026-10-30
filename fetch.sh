#!/bin/sh
# Downloads the two human-annotated sets Darma reads (nothing is generated): UD English EWT (CC BY-SA 4.0) and
# CoarseWSD-20 (CC BY-NC 4.0). Then: cargo run --release -- prepare-data
set -e
mkdir -p data/raw/ud data/raw/coarsewsd
B=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master
for s in train dev test; do curl -s -o data/raw/ud/en_ewt-ud-$s.conllu $B/en_ewt-ud-$s.conllu; done
curl -s -o data/raw/ud/LICENSE.txt $B/LICENSE.txt
C=https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20
curl -s -o data/raw/coarsewsd/LICENSE $C/LICENSE
curl -s -o data/raw/coarsewsd/README.txt $C/README.txt
for w in apple arm bank bass bow chair club crane deck digit hood java mole pitcher pound seal spring square trunk yard; do
  mkdir -p data/raw/coarsewsd/$w
  for f in classes_map.txt train.data.txt train.gold.txt test.data.txt test.gold.txt; do curl -s -o data/raw/coarsewsd/$w/$f $C/$w/$f; done
done
