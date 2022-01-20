#!/usr/bin/python3
from googleapiclient.discovery import build
import argparse
import hashlib
import requests
import shutil
import os
import PIL
from PIL import Image

import workspace as ws

class ImageLoader(ws.Workspace):
    
    MAX_PAGE_START = 190
    
    def __init__(self, workspaces=None, workspace=None, query=None, apik=None, seid=None):
        super().__init__(workspaces=workspaces, workspace=workspace)
        
        assert apik is not None, "API key cannot be none"
        assert seid is not None, "Custom search engine id cannot be null"
        
        self._apik = apik
        self._seid = seid
        
        self._query = query
        
        self._load_existing_images()
            
    def _load_existing_images(self):
        self._existing = []
        self._existing += self._load_image_hashes(self._images)
        self._existing += self._load_image_hashes(self._train)
        self._existing += self._load_image_hashes(self._test)
        
    def _load_image_hashes(self, path):
        duplicates = []
        results = []
        files = os.listdir(path)
        for file in files:
            filepath = os.path.join(self._images, file)
            tokens = file.split(".")
            assert tokens[0] not in results, "Duplicate hash(%s) found in %s" % (tokens[0], path)
            if os.path.isfile(filepath) and tokens[-1] in ["jpg", "jpeg"]:
                # print(file)
                if tokens[0] not in self._existing:
                    results.append(tokens[0])
                else:
                    duplicates.append(tokens[0])
        if 0 < len(duplicates):
            for duplicate in duplicates.copy():
                print("Duplicate hash(%s) found in %s" % (duplicate, path))
                # ck1_filepath = os.path.join(path, "%s.xml" % duplicate)
                # rm_filepath = os.path.join(self._images, "%s.jpg" % duplicate)
                # ck2_filepath = os.path.join(self._images, "%s.xml" % duplicate)
                # if os.path.exists(rm_filepath):
                #     print("Remove duplicate(%s)" % rm_filepath)
                #     print(ck1_filepath, os.path.exists(ck1_filepath))
                #     # os.remove(rm_filepath)
                #     # del duplicates[duplicates.index(duplicate)]
            assert 0 == len(duplicates), "%s Duplicates found in %s" % (len(duplicates), path)
        return results
        
    def search(self, query=None):
        assert query is not None, "Query cannot be null"
        results = []
        resource = build("customsearch", 'v1', developerKey=self._apik).cse()
        start = 0
        while start <= ImageLoader.MAX_PAGE_START:
            result = resource.list(q=query, cx=self._seid, searchType='image', start=start).execute()
            for item in result['items']:
                results.append({'link':item['link'],
                                'format':item['fileFormat'],
                                'metadata':item['image'],
                                'hash':hashlib.sha256(item['link'].encode()).hexdigest()})
            start = result['queries']['nextPage'][0]['startIndex']
        return results
    
    def download(self, results):
        assert results is not None, "Results must not be none"
        print("Downloading %s results" % len(results))
        for result in results:
            print("Process: %s" % result['link'])
            name = "%s.jpg" % result['hash']
            destfile = os.path.join(self._images, name)
            if not os.path.isfile(destfile) and name not in self._existing:
                tmp_file_name = "temp.img"
                try:
                    res = requests.get(result['link'], stream=True, timeout=10)
                    if res.status_code == 200:
                        with open(tmp_file_name,'wb') as f:
                            shutil.copyfileobj(res.raw, f)
                        if "image/jpeg" != format:
                            im = Image.open(tmp_file_name)
                            rgb_im = im.convert('RGB')
                            rgb_im.save(destfile)
                        else:
                            os.rename(tmp_file_name, destfile)
                            print("Success: %s" % hash)
                    else:
                        print("Error: Invalid response (%s)" % res.status_code)
                except requests.exceptions.ReadTimeout as rt:
                    print("Error: Fetch timed out")
                except requests.exceptions.SSLError as ssle:
                    print("Error: SSL")
                except requests.exceptions.ConnectionError as ce:
                    print("Error: Connection")
                except PIL.UnidentifiedImageError as piluie:
                    print("Error: Image format")
                    os.remove("temp.img")
            else:
                print("Skip: Image already exists")
            print()
            
        a = len(self._existing)
        self._load_existing_images()
        print("Added %s images" % (a - len(self._existing)))
        
if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Image search and automated downloader.')
    parser.add_argument('--workspace', help='Path to workspace project')
    parser.add_argument('--query', help='Image query parameter')
    parser.add_argument('--apik', default=None, help='Google API key')
    parser.add_argument('--seid', default=None, help='Google Custom Search Engine ID')
    args = parser.parse_args()
    
    key = None
    if 'GOOGLE_APIK' in os.environ.keys():
        apik = os.environ['GOOGLE_APIK']
    if args.apik is not None:
        apik = args.apik
    
    seid = None
    if 'GOOGLE_SEID' in os.environ.keys():
        seid = os.environ['GOOGLE_SEID']
    if args.seid is not None:
        seid = args.seid
        
    loader = ImageLoader(workspace=args.workspace, apik=apik, seid=seid)
    results = loader.search(query=args.query)
    loader.download(results)

