import json
from glob import glob


if __name__ == '__main__':
    templates = [filename.replace('eloquentarduino/', '')
     for filename in glob('eloquentarduino/templates/**/*.jinja', recursive=True)]

    print(json.dumps(templates).replace('/', '\\/').replace('"', '\\"'))