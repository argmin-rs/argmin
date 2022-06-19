#!/usr/bin/env bash

# Based on a suggestion by @dhardy in https://github.com/rust-lang/mdBook/issues/706

mkdir -p src
cat << EOF > src/lib.rs
#![allow(non_snake_case)]
#[macro_use]
extern crate doc_comment;
EOF

for doc in ../src/*.md
do
    NAME=$(basename $doc .md)
    NAME=${NAME//./_}
    NAME=${NAME//-/_}
    echo -e "doctest\041(\"../$doc\");" > src/$NAME.rs
    echo "mod $NAME;" >> src/lib.rs
done
