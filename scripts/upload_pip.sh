rm -r build dist cpnet.egg-info
python setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
rm -r build dist cpnet.egg-info
