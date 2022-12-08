# AWS create a new instance for install bot

## 1.Set the default login user and root password：

sudo passwd ubuntu

sudo passwd root

## 2.The default user and the root user can log in remotely:

change sshd cofig vim /etc/ssh/sshd_config

change "PasswordAuthentication no" to "PasswordAuthentication yes"

Comment this line："PermitRootLogin prohibit-password"

Add a new line: PermitRootLogin yes 

service sshd restart

## 3.Install bot environment(docker)
### install docker
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun

### install docker-compose
curl -L "https://github.com/docker/compose/releases/download/v2.2.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

### Download the docker-compose file from the repository
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml

### Pull the freqtrade image
docker-compose pull

### Create user directory structure
docker-compose run --rm freqtrade create-userdir --userdir user_data
