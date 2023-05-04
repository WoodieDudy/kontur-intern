#!/bin/bash

file_id="1u4dhGWsXXqH1jP-FLuOJ7lqZtmuqUtDC"
file_name="rubert-base-cased-ner-kontur.ckpt"
destination="weights/"

if [ ! -d "$destination" ]; then
  mkdir "$destination"
fi

if [ -f "$destination$file_name" ]; then
  echo "File $file_name already exists in the $destination directory. No need to download."
else
  gdown "https://drive.google.com/uc?id=$file_id" -O "$destination$file_name"
  echo "File $file_name has been downloaded to the $destination directory."
fi
