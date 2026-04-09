from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ")for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")

setup(
    name="ml project",
    version="0.1",
    author='vinoth G',
    author_email="vinothvinothg1812@mail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
    
)
