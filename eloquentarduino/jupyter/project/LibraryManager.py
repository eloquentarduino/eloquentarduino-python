import re
import requests
import os.path
import pathlib
from io import BytesIO
from zipfile import ZipFile


class LibraryManager:
    """
    Manage project dependencies
    @added 0.1.22
    """
    def __init__(self, project):
        """
        :param project: Project
        """
        self.project = project

    def from_github(self, repo_url, branch='master', tag=None, rewrite_includes=False):
        """
        Download library from Github
        :param repo_url: str
        :param branch: str
        :param tag: str
        :param rewrite_includes: bool if True, replaces includes with absolute paths (experimental)
        """
        archive_url = '%s/archive/refs/%s/%s.zip' % (
            repo_url,
            'tags' if tag is not None else 'heads',
            tag or branch
        )

        response = requests.get(archive_url, stream=True)

        if response.status_code is not 200:
            self.project.logger.error('Cannot download archive from %s', archive_url)
            return

        self.project.logger.info('Downloading library from Github archive %s', archive_url)

        zip = ZipFile(BytesIO(response.content))
        repo_name = repo_url.split('/')[-1]
        repo_qualified_name = '%s-%s' % (repo_name, tag or branch)
        src = '%s/src/' % repo_qualified_name
        fallback_src = '%s/' % repo_qualified_name
        libs_folder = self.project.get_relative_filename('src/')
        files = []

        for file in zip.infolist():
            if (file.filename.startswith(src) or file.filename.startswith(fallback_src)) and not file.filename.endswith('/'):
                # remove nesting
                file.filename = file.filename.replace(src, '').replace(fallback_src, '')
                files.append(file.filename)
                zip.extract(file, libs_folder)

        if rewrite_includes:
            for filename in files:
                needs_rewrite = False
                actual_filename = os.path.join(libs_folder, filename)
                base_folder = os.path.dirname(filename)

                with open(actual_filename, encoding='utf-8') as file:
                    try:
                        contents = file.read()
                    except UnicodeDecodeError:
                        continue

                    for match in re.finditer(r'#include "([^.][^"]+)"', contents):
                        needs_rewrite = True
                        include_path = match.group(1)
                        relative_include = os.path.relpath(include_path, base_folder)

                        if not relative_include.startswith('.'):
                            relative_include = './%s' % relative_include

                        included_file = self.project.get_relative_filename(os.path.join(libs_folder, include_path))

                        if os.path.isfile(included_file):
                            contents = contents.replace(include_path, relative_include)

                if needs_rewrite:
                    with open(actual_filename, 'w', encoding='utf-8') as file:
                        file.write(contents)
