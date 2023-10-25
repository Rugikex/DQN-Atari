import os

# Remove all files in folder models except README.md
for filename in os.listdir(os.path.join(os.getcwd(), 'models')):
    if filename != 'README.md':
        os.remove(os.path.join(os.getcwd(), 'models', filename))

print('Models cleared')
