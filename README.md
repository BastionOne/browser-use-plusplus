### Installation:
## 1. Clone this repo
```
git clone --recurse-submodules git@github.com:BastionOne/browser-use-plusplus.git bupp
```
## 2. Install dependencies (and install our modified version of browser-use)
```
uv pip install . && uv pip install ./browser-use 
```

### Running
## Run an example
```
python cli.py run .bupp\sites\single_component\ashby.json
```
## Run a test
```
python cli.py run_test single_component 3
```
## Run all tests
```
python cli.py run_test
```