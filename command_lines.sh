#To send files: 
scp source_file user@host:destination_file

#source_file: The path to the file or directory you want to send.
#user: The username on the remote machine.
#host: The hostname or IP address of the remote machine.
#destination_file: The path where you want to save the file on the remote machine.


#Send a file to a remote machine:
scp /path/to/local/file.txt user@remote_host:/path/to/destination/

#Send a file to a remote machine with a different name:
scp /path/to/local/file.txt user@remote_host:/path/to/destination/new_file.txt

#Send a directory to a remote machine:
scp -r /path/to/local/directory user@remote_host:/path/to/destination/

#Send multiple files to a remote machine:
scp /path/to/local/file1.txt /path/to/local/file2.txt user@remote_host:/path/to/destination/

#Send a file from a remote machine to your local machine:
scp user@remote_host:/path/to/remote/file.txt /path/to/local/destination/


#Receive a file from a remote machine:
scp user@remote_host:/path/to/remote/file.txt /path/to/local/destination/

#Receive a file from a remote machine with a different name:
scp user@remote_host:/path/to/remote/file.txt /path/to/local/destination/new_file.txt

#Receive a directory from a remote machine:
scp -r user@remote_host:/path/to/remote/directory /path/to/local/destination/
