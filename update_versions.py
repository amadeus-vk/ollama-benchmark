version='0.3'
import os
import re
import zlib
import json

# The file to store CRC checksums
MANIFEST_FILE = 'version_manifest.json'

def load_manifest():
    """Loads the CRC manifest file if it exists."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_manifest(manifest):
    """Saves the updated CRC manifest file."""
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

def update_file_versions():
    """
    Scans files in the current directory, compares their CRC checksums against a
    stored manifest, and increments a version number in the file's header
    if its content has changed.
    """
    manifest = load_manifest()
    
    # Exclude this script and the manifest from being processed
    script_name = os.path.basename(__file__)
    files_to_process = [
        f for f in os.listdir('.') 
        if os.path.isfile(f) and f not in [script_name, MANIFEST_FILE] and not f[0] == '.' 
    ]
    
    manifest_updated = False

    for filename in files_to_process:
        try:
            with open(filename, 'rb') as f:
                content = f.read()
            
            current_crc = zlib.crc32(content)
            previous_crc = manifest.get(filename)

            # Process if CRC has changed or if the file is new to the manifest
            if current_crc != previous_crc:
                lines = content.decode('utf-8', errors='ignore').splitlines()
                
                for i, line in enumerate(lines[:3]): # Check only first 3 lines
                    # match = re.search(r'(version=)(\d+)', line)
                    match = re.search(r'^#?\ ?(version[=:]\ ?)(["\'\ ])([\d.]+)(["\'])', line)
                    if match:
                        prefix = match.group(1)
                        enc1 = match.group(2)
                        versionstr = match.group(3)
                        enc2 = match.group(4)
                        if not '.' in versionstr:
                            current_version = int(versionstr)
                            new_version = current_version + 1
                        else:
                            current_version = versionstr
                            parts = versionstr.split('.')
                            if int(parts[-1]) < 10:
                                parts[-1] = str(int(parts[-1]) + 1)
                            else:
                                parts[-1] = '0'
                                parts[-2] = str(int(parts[-2]) + 1)

                            new_version = '.'.join(parts)
                        
                        # Replace the old version number in the specific line
                        lines[i] = re.sub(r'(version[=:]\ ?)["\'][\d.]+["\']', f"{prefix}{enc1}{new_version}{enc2}", line)
                        
                        # Rewrite the entire file with the updated content
                        with open(filename, 'w', encoding='utf-8', newline='\n') as f_write:
                            f_write.write('\n'.join(lines))
                        
                        print(f"✅ Updated {filename}: version {current_version} -> {new_version}")
                        break # Stop after finding the first version string
                
                with open(filename, 'rb') as f:
                    content = f.read()
                # Update manifest for the changed file
                changed_crc = zlib.crc32(content)
                manifest[filename] = changed_crc
                manifest_updated = True

        except Exception as e:
            print(f"⚠️  Could not process {filename}: {e}")

    if manifest_updated:
        save_manifest(manifest)
        print("\nManifest saved.")
    else:
        print("\nNo file changes detected.")

if __name__ == "__main__":
    update_file_versions()