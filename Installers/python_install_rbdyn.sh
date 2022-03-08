echo "Installing eigen, sva and rbdyn"
# build individually with cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON
# look for the folder that contains setup.py, run pip install .

echo "Installing dependent libraries"
pushd ./
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
git checkout 8.0.0
mkdir -p build 
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=../../temp_install
make -j
make install
popd
rm -rf 

echo "Installing Eigen3ToPython"
pushd ./
git clone https://github.com/jrl-umi3218/Eigen3ToPython.git
cd Eigen3ToPython
git checkout 1.0.2
pip install -r requirements.txt
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON -DCMAKE_INSTALL_PREFIX=../../temp_install
make -j
make install
cd python3
pip install .
popd
rm -rf Eigen3ToPython

echo "Installing SpaceVecAlg"
pushd ./
git clone --recursive https://github.com/jrl-umi3218/SpaceVecAlg
cd SpaceVecAlg
git checkout v1.2.0
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON -DCMAKE_INSTALL_PREFIX=../../temp_install
make -j
make install
cd binding/python/sva/python3
pip install .
popd
rm -rf SpaceVecAlg

echo "Installing RBDyn"
pushd ./
git clone --recursive https://github.com/jrl-umi3218/RBDyn
cd RBDyn
git checkout v1.5.2
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON -DCMAKE_INSTALL_PREFIX=../../temp_install
make -j
make install
cd binding/python/rbdyn/python3
pip install .
popd
rm -rf RBDyn

rm -rf tinyxml2
rm -rf build
rm -rf temp_install
