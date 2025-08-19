import sys
try:
    import fracnetics
    print("✅ import fracnetics") 
except Exception as e:
    print("❌ error importing fracnetics:")
    print(e)
    sys.exit(1)


