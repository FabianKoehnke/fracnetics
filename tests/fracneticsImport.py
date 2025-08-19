import sys
try:
    import fracnetics
    print("import fracnetics successfull") 
except Exception as e:
    print("❌ error importing fracnetics:")
    print(e)
    sys.exit(1)


