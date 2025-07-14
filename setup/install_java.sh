echo "Setting up Java 23.0.1"
wget https://download.oracle.com/java/23/archive/jdk-23.0.1_linux-x64_bin.tar.gz
tar -xvf jdk-23.0.1_linux-x64_bin.tar.gz
rm jdk-23.0.1_linux-x64_bin.tar.gz
echo "export JAVA_HOME=~/jdk-23.0.1" >> ~/.bashrc
echo "export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so" >> ~/.bashrc
# export JAVA_HOME=~/jdk-23.0.1
# export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so