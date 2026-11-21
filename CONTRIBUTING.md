# Contributing to AIMET
We're thrilled that you'd like to contribute to the AIMET project! Your support is essential for keeping this project great and for making it better.

- [Before you begin](#before-you-begin)
- [Guidelines](#guidelines)
- [Branching strategy](#branching-strategy)
- [Setup](#setup)
- [Get code](#get-code)
- [Development](#development)
  * [Build](#build)
  * [Test](#test)
  * [Commit](#commit)
  * [Branch update](#branch-update)
  * [Branch cleanup](#branch-cleanup)
- [Submission](#submission)

## Before you begin
- Please read our [Code of Conduct](CODE-OF-CONDUCT.md) and [License](LICENSE) and ensure that you agree to abide by them.
- For every new feature or bug fix, always start a new issue on https://github.com/qualcomm/aimet/issues.
- To contribute a bug-fix, please follow the steps in the next sections without any further discussion.
- To contribute new features, extensions, utility functions or other significant changes, please describe and discuss the change with us via the GitHub issue that you created above. **A pull request (PR) submitted without discussion and agreement with the project maintainers may be subject to rejection, or significant changes may be requested prior to its acceptance.**

## Guidelines
Please follow the guidelines below to increase the likelihood and speed of your PR acceptance:
- Follow the existing style in the file or folder where possible. We try and adhere to [pep8](https://www.python.org/dev/peps/pep-0008/) for Python and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html) for C++.
- Add new unit tests or update existing tests to verify the code that you contribute.
- Keep your change as focused as possible. If you want to make multiple independent changes, consider submitting them as separate PRs.
- Write a [good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).
- Every commit must be signed with the [Developer Certificate of Origin](https://developercertificate.org) (by adding the `-s` option to your `git commit` command).
- Each PR submission will trigger a build, test, code quality check and static analysis processes. Submitters are required to fix all failures and warnings prior to acceptance.

## Branching strategy
Contributors should develop on [their own fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) on branches based off of the `develop` branch and then pull requests should be made into the [upstream `develop` branch](https://github.com/qualcomm/aimet/tree/develop).

## Setup
Go to https://github.com/qualcomm/aimet and fork the repo using [these instructions](https://help.github.com/en/github/getting-started-with-github/fork-a-repo).

To develop inside Docker, follow the [Requirements](packaging/docker_install.md#requirements) and [Setup](packaging/docker_install.md#setup-the-environment) sections in [docker_install.md](packaging/docker_install.md). To develop directly on your machine, see [Building from source](Docs/overview/install/build_from_source.rst). Then return to this page.

## Get code
[Sync your fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) with the latest from the upstream repository.

Get the aimet code as follows:
```
git clone https://github.com/YOUR_USERNAME/aimet.git
cd aimet
```
*IMPORTANT:* Setup your pre-commit and commit-msg hook using the following way:
```
pip install pre-commit
pre-commit install
ln -s $(realpath -s .githooks/commit-msg) .git/hooks/commit-msg
```

## Development
Start a new issue on https://github.com/qualcomm/aimet/issues.

Create a branch for your feature
```
git checkout -b branch_short_feature_description
```

Now you may begin development. Once your development is complete, please ensure that the code builds successfully and that all tests pass using the instructions in the next sections.

### Build
Build the code using `scripts/all/build_and_test.py` (wheels are written to `wheelhouse/`):
```
cd aimet
python3 scripts/all/build_and_test.py --task build --package aimet_torch   # or aimet_onnx; add --gpu for CUDA
```

### Test
Build (if needed) and run the unit tests:
```
python3 scripts/all/build_and_test.py --task test --package aimet_torch   # or aimet_onnx; add --gpu for CUDA
```

### Commit
Commit the code and checkpoint it on your branch using the following procedure.

To display the files that you modified or added:
```
git status
```

To stage new (untracked) or existing files or folders for commit, do the following for each file or folder name that was added or changed:
```
git add <file or folder name that was added or changed>
```

To commit your changes:
```
git commit -s -m "Commit message"
```
>*IMPORTANT:* The -s option is required during the commit step (DCO signoff).

To push your branch to the remote:
```
git push origin branch_short_feature_description
```

### Branch update
Before merging, it is recommended that you update your branch to the latest on develop using the following steps:
```
git fetch
git checkout develop
git pull origin develop
git checkout branch_short_feature_description
```
Rebase your changes:
```
git rebase develop
```
Fix any conflicts that may arise. Then complete the rebasing procedure as follows:
```
git status

# Run the next 2 commands ONLY IF you needed to fix any conflicts.
# Run this for each file that you changed
git add <file or folder name that was added or changed>
git rebase --continue
```
Re-build the code on your branch and run all tests. Then update your remote branch:
```
git push origin branch_short_feature_description --force-with-lease
```

### Branch cleanup
It is recommended that you commit code to your branches often. Prior to pushing the code and submitting PRs, please try to clean up your branch by squashing multiple commits together and amending commit messages as appropriate. See these pages for details:
https://blog.carbonfive.com/2017/08/28/always-squash-and-rebase-your-git-commits
https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History

## Submission
When you're ready to submit your code, issue a pull request from the branch on your FORK into the develop branch on the upstream repository using these [instructions](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).
1. Go to your forked repo page `https://github.com/YOUR_USERNAME/aimet` and click "New Pull Request".
1. Under "*compare changes*", select the base (destination) repository as `quic/aimet` and the branch as `base:develop` to the left of the arrow.
1. Under "*compare changes*", select the head (source) repository as `YOUR_USERNAME/aimet` and the branch as `base:branch_short_feature_description` to the right of the arrow.
1. Click "*Create Pull Request*" which will initiate the PR and take you to the PR page.
    - In the PR page, click *Reviewers* on the top left and select one. He/she will receive an email notification.
    - In the PR page, click *Assignee* on the top left and select one. This person can be the reviewer or someone else or even the code submitter.
1. Wait for the outcome of the continuous integration (CI) build and test job, and for any review feedback from the project maintainers.
