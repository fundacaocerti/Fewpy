# Contributing

Thanks for taking the time to contribute!

- [First Contribution](#first-contribution)
- [Release a New Version](#release-a-new-version)

# First Contribution

How to make your first contribution:

- [1. Create a GitHub Account](#1-create-a-github-account)
- [2. Find an Issue to Work On](#2-find-an-issue-to-work-on)
- [3. Install Git](#3-install-git)
- [4. Fork the Project](#4-fork-the-project)
- [5. Clone your Fork](#5-clone-your-fork)
- [6. Create a New Branch](#6-create-a-new-branch)
- [7. Run Fewpy Locally](#7-run-fewpy-locally)
- [8. Make your Changes](#8-make-your-changes)
- [9. Commit and push your Changes](#9-commit-and-push-your-changes)
- [10. Add changelog Entries](#10-add-changelog-entries)
- [11. Create a GitHub PR](#11-create-a-github-pr)
- [12. Update your branch if Needed.](#12-update-your-branch-if-needed)

### 1. Create a GitHub Account

Ensure you have a [GitHub account][github-join] and you are logged in.

### 2. Find an Issue to Work With

Visit the [fewpy issues page][brutils-issues] and find an issue that interests you and hasn't been assigned yet.

Leave a comment in the issue with "bora!". A bot will assign the issue to you. Once assigned, proceed to the next step.

Feel free to ask any questions in the issue's page before or during the development process.

When starting to contribute, it is advised to handle one issue at a time. This ensures others have the chance to collaborate and avoids inactive resources.

### 3. Install Git

Make sure you have [Git installed][install-git].

### 4. Fork the Project

[Fork the brutils repository][github-forking].

### 5. Clone your Fork

[Clone your fork][github-cloning] locally.

### 6. Create a New Branch

Go into the brutils folder:

```bash
cd fewpy
```

And create a new branch with the issue number you’re working on:

```bash
git checkout -b <issue_number>
```

Example:

```bash
git checkout -b 386
Switched to a new branch '386'
```

### 7. Run Fewpy

### Requirements

- [Python 3.8+][python]
- [UV][uv]

## Installation with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the dependencies, run:

```bash
uv sync
```

To view how execute and run the fewpy with uv view the
documentation [uv].

### 8. Make your changes

Now it’s time to implement your changes.

Check the issue description for instructions or ideas in the section “Describe alternatives you considered.” Make sure your changes resolve everything mentioned in the issue.

We document our code using [docstrings][docstring-definition]. All modules, classes, functions, and methods should have docstrings. Your changes should reflect updated docstrings, especially if any parameters were changed.

We follow this pattern for docstring consistency:

```python
class Example:
    """
    Explain the purpose of the class

    Attributes:
        x[dict]: Short explanation here
        y[type, optional]: Short explanation here
    """

    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def foobar(self, w):
        """
        Purpose of the function

        Args:
            name[type]: Short explanation here

        Returns:
            type: Short explanation here

        Example:
            >>> command 1
            output 1
            >>> command 2
            output 2
        """
        ...
        return value

```

### 9. Commit and push your changes

Format your code by running the command:

Add your changes to the staging area:

```bash
$ git add --all
```

Commit your changes:

```bash
$ git commit -a -m "<commit_message>"
```

Make the necessary changes and commits and push them when ready.

### 10. Add changelog entries

#### What is a changelog?

A changelog is a file that contains a chronologically organized list of notable changes for each version of a project.

#### Why maintain a changelog?

To make it easier for users and contributors to see exactly what notable changes have been made between each release (or version) of the project.

#### Where is the Fewpy changelog?

The fewpy changelog is available at [CHANGELOG.md][changelog].

#### Guiding principles

- Changelogs are for humans, not machines.
- There should be an entry for every version.
- The same types of changes should be grouped together.
- Versions and sections should be linkable.
- The most recent version comes first.
- The release date of each version is displayed.

#### What justifies an entry in the changelog?

- Security fixes: These should be documented with the type labeled as “security” to alert users to resolved security issues.
  Example: “Fixed a critical vulnerability that allowed remote code execution.”

- User-facing changes: Changes that directly affect how users interact with the software, including new features, changes to existing features, or UI improvements.
  Example: “Added a new filter option on the results page to make searching easier.”

- Significant performance improvements: Should be recorded when they result in noticeable improvements in speed or efficiency that impact the user experience.
  Example: “The home page loading time was reduced by 40% after backend optimization.”

- Changes affecting compatibility: Adjustments to maintain compatibility with other tools, systems, or versions.
  Example: “Updated library X to version 2.0 to support the new version of Python.”

- Changes to the public API: Changes that affect how developers interact with the public API, such as adding new routes or modifying existing ones.
  Example: “Added a new /api/v1/users route for user management.”

- Dependency changes: Updates or changes to the project’s dependencies that may affect software behavior or compatibility.
  Example: “Updated dependency package Y to version 3.1.4, which includes important security fixes.”

#### What should NOT go in the changelog

Although the changelog is a valuable tool for documenting changes, some information should not be included. Here are some examples of what should not appear in the changelog:

- Internal Code Changes: Changes that do not affect the end-user experience, such as internal refactoring that does not alter functionality, need not be documented.
  Example: “Refactored internal functions” or “Fixed inconsistent tests.”

- Non-notable performance improvements: Performance improvements that do not result in noticeable changes or clear benefits for the end user do not need to be included.
  Example: “Optimized internal algorithms.”

- Minor bug fixes: Bug fixes that do not impact the general use or end-user experience can be omitted.
  Example: “Fixed a small typo in the code.”

- Documentation-only changes: Changes that affect only the documentation, without modifying the software behavior, usually don’t need to be included.
  Example: “Updated README.md to reflect new dependencies.”

- Excessive technical details: Overly technical information that is irrelevant to the end user or does not provide context on the impact of the change should be avoided.
  Example: “Changed memory management in class X.”

- Maintainer entries: Changes related only to the development or internal maintenance process, such as CI/CD tool configuration adjustments, are generally not relevant for the changelog.
  Example: “Updated GitHub Actions configuration.”

- A bug introduced and fixed in the same release does not need a changelog entry.

Avoid including this information to keep the changelog focused and useful for project users and contributors.

#### How to add an entry to the changelog

The changelog is available in the [CHANGELOG.md][changelog] file.

First, you need to identify the type of your change. Types of changes:

- `Added` for new features.
- `Changed` for changes to existing features.
- `Deprecated` for features that will soon be removed.
- `Fixed` for any bug fixes.
- `Removed` for features that were removed.
- `Security` in case of vulnerabilities.

You should always add new entries to the changelog in the `Unreleased` section. At the time of release, we will move the changes from the `Unreleased` section to a new version section.

So, within the `Unreleased` section, you should add your entry to the appropriate section by type. If there is no section yet for the type of your change, you should add one.

Let’s see some examples. Suppose you have a new `Fixed` change to add, and the current CHANGELOG.md file looks like this:

```md
## [Unreleased]

### Added

- Utility `get_address_from_cep` [#358](https://github.com/fundacaocerti/Fewpy/pull/358)

### Changed

- Utility `fmt_voter_id` renamed to `format_voter_id` [#221](https://github.com/fundacaocerti/Fewpy/issues/221)
```

You would need to add a new `Fixed` section and include your new entry there:

```md
## [Unreleased]

### Added

- Utility `get_address_from_cep` [#358](https://github.com/fundacaocerti/Fewpy/pull/358)

### Changed

- Utility `fmt_voter_id` renamed to `format_voter_id` [#221](https://github.com/fundacaocerti/Fewpy/issues/221)

### Fixed

- My changelog message here. [#<issue_number>](issue_link)
```

Note that the order of sections by type matters. We have a lint that checks this, so the sections must be ordered alphabetically. First `Added`, then `Changed`, third `Deprecated`, and so on.

Now, let’s say you have another entry to add, and its type is `Added`. Since we already have a section for that, you should just add a new line:

```md
## [Unreleased]

### Added

- Utility `get_address_from_cep` [#358](https://github.com/fundacaocerti/Fewpy/pull/358)
- My other changelog message here. [#<issue_number>](issue_link)

### Changed

- Utility `fmt_voter_id` renamed to `format_voter_id` [#221](https://github.com/fundacaocerti/Fewpy/issues/221)

### Fixed

- My changelog message here. [#<issue_number>](issue_link)
```

This content is based on the [Keep a Changelog][keep-a-changelog] site, as we follow its guidelines.

### 10. Create a GitHub PR

[Create a GitHub PR][github-creating-a-pr] to submit your changes for review. To ensure your Pull Request (PR) is clear, effective, and reviewed quickly, follow these best practices:

#### Write a Descriptive PR Title

- Use clear and specific titles to describe the purpose of your changes. A good title helps maintainers understand the PR’s intent at a glance and improves project traceability.
- **Example**: Instead of “Fix issue,” use “Add utility `convert_uf_to_text` to handle Brazilian state codes.”
- **Benefits**:
  - Clear titles make it easier for reviewers to prioritize and understand the PR.
  - They improve the project’s organization and searchability.

#### Provide a Detailed PR Description

- Include a comprehensive description in your PR to explain:
  - **What** was done (e.g., added a new function, fixed a bug).
  - **Why** it was done (e.g., to address a specific issue or improve performance).
  - **What issues** were resolved or improvements applied (e.g., link to the issue or describe the enhancement).
- **Example**:
  This PR adds the convert_uf_to_text utility to convert Brazilian state codes (e.g., "SP") to full state names (e.g., "São Paulo"). It addresses issue #474 by improving code reusability for address formatting. The function includes input validation and updated tests.
- **Benefits**:
- Detailed descriptions speed up the review process by providing context.
- They help future maintainers understand the code’s purpose and history.

#### Link the PR to the Related Issue

- Reference the issue your PR addresses using keywords like `Closes #474` or `Fixes #474` in the PR description. This automatically closes the issue when the PR is merged.
- **Example**: `Closes #474`
- **Benefits**:
- Linking issues keeps the repository organized and ensures tasks are tracked.
- It automates issue closure, reducing manual work for maintainers.
- For more details, see the [GitHub documentation on closing issues automatically](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).

#### Verify the PR Description Template

- Ensure your PR follows the repository’s PR description template (if provided). Check all required items, such as test coverage, documentation updates, or changelog entries.
- **Example Checklist**: (showing how it looks when completed):
- [x] Code changes are tested.
- [x] Documentation (READMEs) is updated.
- [ ] Changelog entry is added.
- **Syntax Note**:
- Use [x] to mark completed items and [ ] for incomplete ones, with no spaces inside the brackets (e.g., [ x ] or [x ] will not render correctly on GitHub).
- **Benefits**:
- Adhering to the template ensures your PR is complete and ready for review.
- It reduces back-and-forth with reviewers, speeding up the merge process.

### 11. Update your branch if needed.

[Ensure your branch is up to date with main][github-sync-pr].

# Release a New Version

Here you will find how to deploy a new production version of brutils:

- [1. Create a Release Issue](#1-create-a-release-issue)
- [2. Create a Release PR](#2-create-a-release-pr)
- [3. Deploy via GitHub](#3-deploy-via-github)

### 1. Create a Release Issue

#### Create the Issue

To create the issue, you can use the feature template, naming the issue Release v<version>. [Example](https://github.com/fundacaocerti/Fewpy/issues/322)

#### Create a Branch

The name of the branch created for the release is related to the Issue number, as shown in [this example](https://github.com/fundacaocerti/Fewpy/pull/326)

### 2. Create a Release PR

#### Update the Library Version

Increment the version number, following the [Semantic Versioning][semantic-versioning],
in the `pyproject.toml` file: https://github.com/fundacaocerti/Fewpy/blob/main/pyproject.toml#L3

#### Update the CHANGELOG.md

Add a title for the new version with the new number and the current date, as seen in [this example](https://github.com/fundacaocerti/Fewpy/blob/main/CHANGELOG.md?plain=1#L9).

And add the version links, like [this example](https://github.com/fundacaocerti/Fewpy/blob/eac770e8b213532d2bb5948d117f6f4684f65be2/CHANGELOG.md?plain=1#L76)

#### Create the PR

Create a PR named `Release v<version>` containing the two changes mentioned above. In the description of the Pull Request, add the modified section of the changelog.

[Example of Release PR](https://github.com/fundacaocerti/Fewpy/releases/pull/128)

#### Merge the PR

Once the PR is accepted and passes all checks, merge it.

### 2. Deploy via GitHub

The new version release in production is done automatically when a [new release is created][creating-releases] on GitHub.

- Fill in the `tag version` should be `v<version>` (e.g `v2.0.0`).
- Fill in the `release title` should be the same as tag version (e.g `v2.0.0`).
- Fill in the `release description` should be the content copied from the CHANGELOG.md file from the
  corresponding version section.

Real examples are available at: https://github.com/fundacaocerti/Fewpy/releases

When the GitHub deployment is completed, the new version will also be automatically released on
[PyPI][brutils-on-pypi]. Download the new brutils version from PyPI and test to ensure everything is working as expected.

[fewpy-issues]: https://github.com/fundacaocerti/Fewpy/issues
[fewpy-on-pypi]: https://pypi.org/project/fewpy/
[fewpy-releases]: https://github.com/fundacaocerti/Fewpy/releases
[changelog]: https://github.com/fundacaocerti/Fewpy/releases
[creating-releases]: https://docs.github.com/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release
[docstring-definition]: https://www.python.org/dev/peps/pep-0257/#what-is-a-docstring
[github-cloning]: https://docs.github.com/repositories/creating-and-managing-repositories/cloning-a-repository
[github-creating-a-pr]: https://docs.github.com/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request
[github-forking]: https://docs.github.com/get-started/quickstart/contributing-to-projects
[github-join]: https://github.com/join
[github-sync-pr]: https://docs.github.com/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/keeping-your-pull-request-in-sync-with-the-base-branch
[keep-a-changelog]: https://keepachangelog.com/en/1.0.0/
[uv]: https://docs.astral.sh/uv/
[python]: https://www.python.org/downloads/
[release-pr-example]: https://github.com/fundacaocerti/Fewpy/pull/326
[semantic-versioning]: https://semver.org
