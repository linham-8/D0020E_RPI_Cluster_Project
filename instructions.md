# Introduction
This guide goes through all of the steps necessary to setup a raspberry pi-cluster. One of the Pis will act as a head node and the rest as compute nodes. The compute nodes network boot from the SSD, which is also used as shared storage. For the most part, this guide follows the guide linked below, with modifications made where necessary:
https://www.raspberrypi.com/tutorials/cluster-raspberry-pi-tutorial/

## Hardware
- 5 x Raspberry Pi 4 Model B Rev 1.1
- 5 x Raspberry Pi PoE+ HAT
- 5 x Fans
- 5 x 16-32 GB SD cards (Only 1 necessary)
- 7 x Ethernet cables
- A gigabit router with an ethernet-connection
- A USB-to-Gigabit ethernet adapter
- A 1TB SATA SSD with a SATA to USB connector
- An 8-port Gigabit PoE Switch
- A 3D-printed case for the Raspberry Pis

## Steps

### Raspberry Pi OS installation
Install the Raspberry Pi Imager on a computer and connect an SD card to it.  
Start the imager program and select Raspberry Pi 4 as device, Raspberry Pi OS Lite (64-bit) as the OS and the SD card as storage.  
When prompted to apply OS customisation settings, select edit settings.  
Under General, set the hostname, username and password and locale settings appropriately. Example hostnames:
- cluster (head-node)
- pi1
- pi2
- pi3
- pi4  
  
Under Services, enable SSH.  
Click save, then yes, and run the imaging process.
Repeat the process for each of the remaining SD cards.

### Hardware
Insert the SD cards into the head and compute nodes according to their hostnames, as in the image.  
Mount a fan and Raspberry Pi PoE hat on each of the Raspberry Pis, then mount them in the case.  
Connect the router to the internet.  
Connect an ethernet cable to one of the LAN ports on the router, and the other end to the USB-to-Gigabit ethernet adapter, which then goes into a USB port on the cluster head-node.  
Connect the SATA SSD to one of the other USB ports on the head-node.  
Finally, connect each Raspberry PI to the switch with the ethernet cables. 
  
**Hardware diagram**
![alt text](assets/hardware-wiring.png)
Original image credit: https://www.raspberrypi.com/tutorials/cluster-raspberry-pi-tutorial/#what-were-going-to-build

### Software
#### Connecting to the head node
Begin by going to your browsers clients/DHCP interface. The head-node should have been issued a local IP. Connect to the head-node by running the following ssh command, assuming the username is cluster and IP is 192.168.0.101:  
```bash
ssh cluster@192.168.0.101
```
#### Configure network and DHCP-server
Initially, only the head-node will be assigned an IP. Running nmcli will show the active wired connection to the router, with subnet 192.168.0.0/24:
```bash
$ nmcli
eth1: connected to Wired connection 2
        "ASIX AX88179"
        ethernet (cdc_ncm), 00:24:9B:8A:48:33, hw, mtu 1500
        ip4 default
        inet4 192.168.0.101/24
        route4 192.168.0.0/24 metric 100
        route4 default via 192.168.0.1 metric 100
        inet6 fe80::963b:89ed:5ffe:5280/64
        route6 fe80::/64 metric 1024
```
To access the other Pis in the cluster, the head node needs to be turned into a DHCP server that will assign an IP to each of them as well as the switch. To do this, add a second wired connection called "Wired connection 1" with subnet 192.168.50.1/24 by running the following commands:
```bash
$ sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.50.1/24 ipv4.method manual
$ sudo nmcli con down "Wired connection 1"
$ sudo nmcli con up "Wired connection 1"
```
Running nmcli shows that the connection has been added as *eth0*:
```bash
$ nmcli
eth1: connected to Wired connection 2
        "ASIX AX88179"
        ethernet (cdc_ncm), 00:24:9B:8A:48:33, hw, mtu 1500
        ip4 default
        inet4 192.168.0.101/24
        route4 192.168.0.0/24 metric 100
        route4 default via 192.168.0.1 metric 100
        inet6 fe80::963b:89ed:5ffe:5280/64
        route6 fe80::/64 metric 1024

eth0: connected to Wired connection 1
        "eth0"
        ethernet (bcmgenet), DC:A6:32:18:75:AA, hw, mtu 1500
        inet4 192.168.50.1/24
        route4 192.168.50.0/24 metric 101
        inet6 fe80::6b32:5d6d:c68b:8a4d/64
        route6 fe80::/64 metric 1024
```
To turn the head-node into a DHCP-server, start by installing the DHCP server:
```bash
$ sudo apt install isc-dhcp-server
```
Then edit /etc/dhcp/dhcpd.conf as follows:
```conf
ddns-update-style none;
authoritative;
log-facility local7;

# No service will be given on this subnet
subnet 10.3.31.0 netmask 255.255.255.0 {
}

# The internal cluster network
subnet 192.168.50.0 netmask 255.255.255.0 {
    range 192.168.50.20 192.168.50.250; # IP range used for DHCP
    option routers 192.168.50.1;
    option broadcast-address 192.168.50.255;
    option domain-name "cluster";
    option domain-name-servers 8.8.8.8, 8.8.4.4;
    default-lease-time 600;
    max-lease-time 7200;
}

# Head Node reservation (must be global)
host cluster {
    hardware ethernet dc:a6:32:18:75:aa; # Replace with your eth0 MAC
    fixed-address 192.168.50.1;
}

# Netgear Switch reservation (must be global)
host switch {
    hardware ethernet c8:78:7d:83:8f:90; # Replace with switch MAC
    fixed-address 192.168.50.254;
}
```
Edit /etc/default/isc-dhcp-server as follows:
```conf
INTERFACESv4="eth0"
DHCPDv4_CONF=/etc/dhcp/dhcpd.conf
DHCPDv4_PID=/var/run/dhcpd.pid
```
Edit /etc/hosts as follows:
```conf
127.0.0.1       localhost
::1             localhost ip6-localhost ip6-loopback
ff02::1         ip6-allnodes
ff02::2         ip6-allrouters

127.0.1.1       cluster
127.0.0.1       cluster
192.168.50.1    cluster

192.168.50.254  switch
```
Reboot the head-node to start the DHCP service. Once rebooted, the head-node should have assigned IPs to the compute nodes. Run the following command to check:
```bash
cluster@192.168.0.101:~ $ dhcp-lease-list
To get manufacturer names please download http://standards-oui.ieee.org/oui.txt to /usr/local/etc/oui.txt
Reading leases from /var/lib/dhcp/dhcpd.leases
MAC                IP              hostname       valid until         manufacturer
===============================================================================================
2c:cf:67:64:7c:84  192.168.50.21   pi3            2025-11-27 14:04:40 -NA-  
2c:cf:67:64:7c:cf  192.168.50.23   pi2            2025-11-27 14:02:34 -NA-  
2c:cf:67:64:7d:03  192.168.50.22   pi4            2025-11-27 14:00:35 -NA-  
```
FIX alla behöver vara med
#### Add the external SSD
Network booting requires a bit more space. Begin by formatting and creating an ext4 partition on the SSD. To do this, use gparted or run the following commands:
```bash
$ sudo parted -s /dev/sda mklabel gpt
$ sudo parted --a optimal /dev/sda mkpart primary ext4 0% 100%
$ sudo mkfs -t ext4 /dev/sda1
mke2fs 1.46.2 (28-Feb-2021)
Creating filesystem with 244175218 4k blocks and 61046784 inodes
.
.
.
Allocating group tables: done
Writing inode tables: done
Creating journal (262144 blocks): done
Writing superblocks and filesystem accounting information: done
$
```
Mount the disk to check it can be written to and read from:
```bash
$ sudo mkdir /mnt/usb
$ sudo mount /dev/sda1 /mnt/usb
$ sudo systemctl daemon-reload
```
Edit /etc/fstab so that it is automatically mounted on boot if the manual mount worked, by adding this line:
```conf
/dev/sda1 /mnt/usb auto defaults,user 0 1
```
#### Make the SSD available to the cluster
To share the SSD over the network, install the NFS server software
```bash
sudo apt install nfs-kernel-server
```
Create a mount point
```bash
$ sudo mkdir /mnt/usb/scratch
$ sudo chown pi:pi /mnt/usb/scratch
$ sudo ln -s /mnt/usb/scratch /scratch
```
Edit /etc/exports to list IP addresses which should be able to mount the SSD:
```conf
/mnt/usb/scratch 192.168.50.0/24(rw,sync)
```
Enable and start the rpcbind and nfs-server services:
```bash
$ sudo systemctl enable rpcbind.service
$ sudo systemctl start rpcbind.service
$ sudo systemctl enable nfs-server.service
$ sudo systemctl start nfs-server.service
```
Reboot  
Then, on the compute nodes once network boot has been set up, install nfs-common:
```bash
sudo apt install nfs-common
```
#### Add the first compute node
To network boot our compute nodes, the boot order must first be changed which requires them to boot from the SD card first. SSH into one of them
```bash
cluster@192.168.0.101:~ $ dhcp-lease-list
To get manufacturer names please download http://standards-oui.ieee.org/oui.txt to /usr/local/etc/oui.txt
Reading leases from /var/lib/dhcp/dhcpd.leases
MAC                IP              hostname       valid until         manufacturer
===============================================================================================
2c:cf:67:64:7c:84  192.168.50.21   pi3            2025-11-27 14:04:40 -NA-  
2c:cf:67:64:7c:cf  192.168.50.23   pi2            2025-11-27 14:02:34 -NA-  
2c:cf:67:64:7d:03  192.168.50.22   pi4            2025-11-27 14:00:35 -NA-  
cluster@192.168.0.101:~ $ ssh pi2@192.168.50.23
pi2@192.168.50.23:~ $ sudo raspi-config
```
FIX använd pi1  
Choose Advanced Options > Boot Order > Network Boot, then reboot.  
Once it has rebooted, check that BOOT_ORDER is 0xf21 which means it will boot from SD first, followed by network boot. Then, take note of the ethernet MAC address and serial number of the raspberry pi:
```bash
pi2@192.168.50.23:~ $ ethtool -P eth0
Permanent address: 2c:cf:67:64:7c:cf
cluster@cluster:~ $ grep Serial /proc/cpuinfo | cut -d ' ' -f 2 | cut -c 9-16
bd86fa18
```
Shut down the board and remove the SD card.
#### Set up head node as boot server
Install TFTP server and create a mount point for it to enable the head node to act as a boot server:
```bash
$ sudo apt install tftpd-hpa
$ sudo apt install kpartx
$ sudo mkdir /mnt/usb/tftpboot
$ sudo chown tftp:tftp /mnt/usb/tftpboot
```
Edit /etc/default/tftpd-hpa as follows:
```conf
# /etc/default/tftpd-hpa

TFTP_USERNAME="tftp"
TFTP_DIRECTORY="/mnt/usb/tftpboot"
TFTP_ADDRESS=":69"
TFTP_OPTIONS="--secure --create"
```
Restart the service:
```bash
$ sudo systemctl restart tftpd-hpa
```
One boot image is needed for each compute node. The following commands prepare the image to be used by the first compute node:
```bash
$ sudo su
$ mkdir /tmp/image
$ cd /tmp/image
$ wget -O raspios_lite_latest.img.xz https://downloads.raspberrypi.com/raspios_lite_arm64_latest
$ xz -d raspios_lite_latest.img.xz # Check below on error!
$ kpartx -a -v *.img
$ mkdir bootmnt
$ mkdir rootmnt
$ mount /dev/mapper/loop0p1 bootmnt/ # Check below before running!
$ mount /dev/mapper/loop0p2 rootmnt/
$ mkdir -p /mnt/usb/pi1
$ mkdir -p /mnt/usb/tftpboot/SN # SN is the serial number of the node 
$ cp -a rootmnt/* /mnt/usb/pi1
$ cp -a bootmnt/* /mnt/usb/pi1/boot/firmware
```
If the xz command fails, it could be due to the tmp folder being too small, run the following command:
```bash
$ sudo mount -o remount,size=3G /tmp
```
If /dev/mapper/loopxx does not exist, run:
```bash
$ sudo kpartx -av raspios_lite_latest.img.xz
```
Now we can customise the root file system:
```bash
$ touch /mnt/usb/pi1/boot/firmware/ssh
$ echo pi:$(echo 'raspberry' | openssl passwd -6 -stdin) > /mnt/usb/pi1/boot/firmware/userconf.txt
$ sed -i /UUID/d /mnt/usb/pi1/etc/fstab
$ echo "192.168.50.1:/mnt/usb/tftpboot/6a5ef8b0 /boot/firmware nfs defaults,vers=3 0 0" >> /mnt/usb/pi1/etc/fstab
$ echo "console=serial0,115200 console=tty root=/dev/nfs nfsroot=192.168.50.1:/mnt/usb/pi1,vers=3 rw ip=dhcp rootwait" > /mnt/usb/pi1/boot/firmware/cmdline.txt
```
Add it to /etc/exports on the head node:
```bash
$ echo "/mnt/usb/pi1 192.168.50.0/24(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
```
Cleanup:
```bash
$ systemctl restart rpcbind
$ systemctl restart nfs-server
$ umount bootmnt/
$ umount rootmnt/
$ cd /tmp; rm -rf image
$ exit
```
Finally, edit /etc/dhcp/dhcpd.conf as follows:
```conf
ddns-update-style none;
authoritative;
log-facility local7;
option option-43 code 43 = text;
option option-66 code 66 = text;

# No service will be given on this subnet
subnet 10.3.31.0 netmask 255.255.255.0 {
}

# The internal cluster network
group {
   option broadcast-address 192.168.50.255;
   option routers 192.168.50.1;
   default-lease-time 600;
   max-lease-time 7200;
   option domain-name "cluster";
   option domain-name-servers 8.8.8.8, 8.8.4.4;
   subnet 192.168.50.0 netmask 255.255.255.0 {
      range 192.168.50.20 192.168.50.250;

      # Head Node
      host cluster {
         hardware ethernet dc:a6:32:18:75:aa;
         fixed-address 192.168.50.1;
      }

      # NETGEAR Switch
      host switch {
         hardware ethernet c8:78:7d:b3:bf:90;
         fixed-address 192.168.50.254;
      }

      host pi1 {
         option root-path "/mnt/usb/tftpboot/";
         hardware ethernet dc:a6:32:36:68:61;
         option option-43 "Raspberry Pi Boot";
         option option-66 "192.168.50.1";
         next-server 192.168.50.1;
         fixed-address 192.168.50.11;
         option host-name "pi1";
      }

   }
}
```
Finish by rebooting
```bash
$ sudo reboot
```
#### Network boot the node
If the compute node reboots correctly, it should come back up and be accesible on ssh:
```bash
$ ssh pi@192.168.50.11
```
To prevent the raspberry pi from trying to resize its filesystem on the first boot and also uninstall the swap daemon, run the following commands:
```bash
$ sudo systemctl disable resize2fs_once.service
$ sudo systemctl disable sshswitch.service
$ sudo apt remove dphys-swapfile
```
Now change the hostname by running:
```bash
$ sudo raspi-config
```
and going to System Options > Hostname. Then select Yes to reboot.  
Finally, edit /etc/hosts to allow usage of hostname instead of ip every time:
```conf
127.0.0.1       localhost
::1             localhost ip6-localhost ip6-loopback
ff02::1         ip6-allnodes
ff02::2         ip6-allrouters

127.0.1.1       cluster

192.168.50.1    cluster
192.168.50.254  switch

192.168.50.11   pi1
192.168.50.12   pi2
192.168.50.13   pi3
192.168.50.14   pi4
```
And reboot.
#### Mount the scratch disk
Begin by creating a mount point for the scratch disk:
```bash
$ sudo mkdir /scratch
$ sudo chown pi:pi scratch
```
Add the following line to /etc/fstab:
```conf
192.168.50.1:/mnt/usb/scratch /scratch nfs defaults 0 0
```
Then reboot.

#### Secure shell without a password


#### 




#### LAN & Internet access
Uncomment the following line in /etc/sysctl.conf:
```conf
net.ipv4.ip_forward=1
```
Then configure the iptables:
```bash
$ sudo apt install iptables
$ sudo iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE
$ sudo iptables -A FORWARD -i eth1 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
$ sudo iptables -A FORWARD -i eth0 -o eth1 -j ACCEPT
$ sudo sh -c "iptables-save > /etc/iptables.ipv4.nat"
```
and then add the following line, just above the exit 0 line, in the /etc/rc.local file to load the tables on boot:
```conf
_IP=$(hostname -I) || true
if
[ "$_IP" ]; then
  printf "My IP address is %s\n" "$_IP"
fi

iptables-restore < /etc/iptables.ipv4.nat

exit 0
```
Then reboot.

Try to ping a DNS provider to ensure the internet connection is working:
```bash
$ ping 1.1.1.1
```
If the nodes don't have internet access, the following file may need to be made executable:
```bash
$ sudo chmod +x /etc/rc.local
```
Also, ensure that the NAT table contains masquerade by running the following command:
```bash
$ sudo iptables -t nat -L -n -v
```
Then reboot and try again.

#### Add the next compute node
To enable network boot for the next board, list the pis using `dhcp-lease-list` again, remove the known_hosts file, connectto the compute node and enter raspi-config to enable network boot as previously. Then reboot it:
```bash
$ rm /home/pi/.ssh/known_hosts
$ ssh <username>@129.168.50.21
$ sudo raspi-config
$ sudo reboot
```
Check that the BOOT_ORDER is 0xf21 by running vcgencmd:
```bash
$ vcgencmd bootloader_config
BOOT_UART=0
WAKE_ON_GPIO=1
POWER_OFF_ON_HALT=0


[all]
BOOT_ORDER=0xf21
$
```
Take note of the ethernet MAC address and serial number of the raspberry pi:
```bash
pi@pi1:~ $ ethtool -P eth0
Permanent address: dc:a6:32:36:68:61
pi@pi1:~ $ grep Serial /proc/cpuinfo | cut -d ' ' -f 2 | cut -c 9-16
5644be38
```
Then shut the board down and remove the SD card.
On the head node, the already configured image can be used again:
```bash
$ sudo su
$ mkdir -p /mnt/usb/pi2
$ cp -a /mnt/usb/pi1/* /mnt/usb/pi2
$ mkdir -p /mnt/usb/tftpboot/54e91338
$ echo "/mnt/usb/pi2/boot/firmware /mnt/usb/tftpboot/54e91338 none defaults,bind 0 0" >> /etc/fstab
$ echo "/mnt/usb/pi2 192.168.50.0/24(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
$ exit
$
```
Then edit /mnt/usb/pi2/boot/firmware/cmdline.txt, and replace pi1 with pi2:
```bash
console=serial0,115200 console=tty root=/dev/nfs nfsroot=192.168.50.1:/mnt/usb/rp2,vers=3 rw ip=dhcp rootwait
```
and similarly for /mnt/usb/rpi2/etc/hostname:
```bash
rpi2
```
Then edit the /etc/dhcp/dhcpd.conf file:
```bash
host pi1 {
         option root-path "/mnt/usb/tftpboot/";
         hardware ethernet dc:a6:32:36:68:61;
         option option-43 "Raspberry Pi Boot";
         option option-66 "192.168.50.1";
         next-server 192.168.50.1;
         fixed-address 192.168.50.12;
         option host-name "pi2";
      }
```
Then reboot.  
Both should now be up and running, run the following commands to scan with nmap:
```bash
$ sudo apt install nmap
$ nmap 192.168.50.0/24
FIX lägg till output
```
#### Add the rest of the nodes
Repeat the above steps for the remaining compute nodes, substituting the appropriate MAC address, serial number, and hostname for each of them.

#### Simultaneous control
pssh allows simultaneous control of the cluster nodes and also installs multiple command-line tools. Install it with the following command:
```bash
$ apt install pssh
```
Create a host file listing all compute nodes in the home directory:
```bash
$ cat .pssh_hosts
pi1
pi2
pi3
pi4
```
Test it with the following command:
```bash
$ parallel-ssh -i -h .pssh_hosts free -h
```
