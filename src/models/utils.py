import hashlib

def pid_to_port(pid, base_port=1024, max_port=65535):
    hash_value = int(hashlib.md5(str(pid).encode()).hexdigest(), 16)
    return base_port + (hash_value % (max_port - base_port))
