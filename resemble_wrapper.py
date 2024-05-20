from resemble import Resemble
import requests
import os
from playsound import playsound

def download_wav(url, out_path):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(out_path, 'wb') as f:
            f.write(response.content)
        print("Audio saved to", os.path.abspath(out_path))
    else:
        print("Failed to download. Status code:", response.status_code)
        
def play_wav(path):
    try:
        playsound(path)
    except:
        print("Could not play audio from", os.path.abspath(path))
        
        
class Resemble_Wrapper(object):
    def __init__(self, api_key):
        try:
            Resemble.api_key(api_key=api_key)
        except:
            print("Error authenticating with Resemble. Please check if your API key is correct")
            
    def list_projects(self):
        i = 0
        while i < 5:
            try:
                response = Resemble.v2.projects.all(page=1, page_size=1000)
                if "items" in response:
                    break
            except:
                pass
            i += 1
        else:
            raise Exception("Resemble list projects failed after 5 retries")
        
        out = []
        for project in response["items"]:
            out.append({"name": project["name"],
                        "uuid": project["uuid"],
                        "description": project["description"]})
        
        return out
    
    def list_voices(self):
        i = 0
        while i < 5:
            try:
                response = Resemble.v2.voices.all(page=1, page_size=1000)
                print(response)
                if "items" in response:
                    break
            except:
                pass
            i += 1
        else:
            raise Exception("Resemble list voices failed after 5 retries")
        
        out = []
        for voice in response["items"]:
            out.append({"name": voice["name"],
                        "uuid": voice["uuid"]})        
        return out
    
    def generate(self, title, text, project_uuid, voice_uuid, out_dir="./output/"):
        i = 0
        while i < 5:
            try:
                response = Resemble.v2.clips.create_sync(project_uuid=project_uuid, voice_uuid=voice_uuid, body=text, title=title)
                if "item" in response:
                    break
            except:
                pass
            i += 1
        else:
            raise Exception("Resemble generate clip failed after 5 retries")
        url = response['item']['audio_src']
        
        out_path = os.path.join(out_dir, title + ".wav")
        download_wav(url, out_path)
        
        return out_path
        
        
    