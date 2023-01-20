+++
title = "argmin{}"


# The homepage contents
[extra]
lead = 'Mathematical optimization in pure Rust.'
url = "/docs/getting-started/introduction/"
url_button = "Get started"
crates_url = "https://www.argmin-rs.org/book/"
crates_url_button = "Book"
docs_url = "https://docs.rs/argmin"
docs_url_button = "Docs"
librs_url = "https://lib.rs/argmin"
librs_url_button = "View on lib.rs"
github_url = "https://github.com/argmin-rs/argmin"
github_url_button = "Code"
repo_version = "View on Github"
repo_license = "License: MIT/Apache-2.0"
repo_url = "https://github.com/argmin-rs/argmin"

# Menu items
# [[extra.menu.main]]
# name = "Getting started"
# section = "docs"
# url = "/docs/getting-started/introduction/"
# weight = 10
# 
# [[extra.menu.main]]
# name = "Book"
# section = "docs"
# url = "/docs/getting-started/introduction/"
# weight = 10
#
# [[extra.menu.main]]
# name = "docs.rs"
# section = "docs"
# url = "https://docs.rs/argmin"
# weight = 10

[[extra.menu.main]]
name = "Blog"
section = "blog"
url = "/blog/"
weight = 20

[[extra.list]]
title = "Pure Rust 🦀"
content = 'We aim at providing a wide range of solvers implemented in pure Rust.'

[[extra.list]]
title = "Consistent interface 🔌"
content = 'Easily plug your optimization problem in different solvers thanks to a consistent interface.'

[[extra.list]]
title = "Type agnostic 🔧"
content = "Represent your variables, gradients, Jacobians and Hessians using your own types."

[[extra.list]]
title = "Various math backends 🧮"
content = 'Use Vecs, <a href="https://crates.io/crates/ndarray">ndarray</a>, or <a href="https://crates.io/crates/nalgebra">nalgebra</a> for behind-the-scenes math. Or implement your own backend.'

[[extra.list]]
title = "Observers 👀"
content = "Observe the progress of your optimization by logging to screen or file or implement your own observer."

[[extra.list]]
title = "Checkpointing 🚩"
content = "Mitigate the negative effects of crashes in unstable computing environments by saving regular checkpoints."

[[extra.list]]
title = "Extensible 🪡"
content = "Most features can be exchanged for user supplied implementations thanks to Rusts powerful generics and traits."

[[extra.list]]
title = "Development framework 🏗"
content = "Use argmins tools and ecosystem to implement your own solver -- with all the features mentioned above."

+++
