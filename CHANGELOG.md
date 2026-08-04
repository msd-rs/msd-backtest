# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.6] - 2026-08-04

### Improve

- Improve select strategy building experience

## [0.1.5] - 2026-07-27

### Fix

- Use utf8 encoding when read/write files
- Excel writer ignore INF/NAN, avoid exception

## [0.1.4] - 2026-07-22

### Add

- `SelectorDataProvider.load_snapshot`  to speed up data loading.


## [0.1.3] - 2026-07-17

### Add

- selector module to select stocks
- a new skill named `write-selector` to guide AI to write selector codes


### Changed

- Use polars instead of pandas for better performance


## [0.1.2] - 2026-07-14

### Fix

- add stock shares aware

## [0.1.1] - 2026-07-10

### Add

- More docs

### Fix

- some typo 

## [0.1.0] - 2026-07-10

Initial release.