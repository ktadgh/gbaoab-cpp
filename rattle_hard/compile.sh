python setup.py clean --all
rm -rf build/ *.so
python setup.py install
# ldd rattle_hard_cuda*.so | grep cusolver
