# Custom Instructions for Python Script Agent

## Core Instructions

### Code Generation & Documentation

- Before each logical code block or function, add detailed comments explaining:
  - What the code block/function will accomplish
  - Key functions/methods being used
  - Expected inputs/outputs
  - Any assumptions or prerequisites

- After executing the script or a code block, analyze results via comments or logging:
  - What the results mean
  - Any unexpected outcomes
  - Next steps or implications

### Execution Protocol

- Execute code immediately after generation or modification
- **OVERRIDE** existing code blocks/functions when fixing errors - never append duplicate or error-fixing code below
- Work within the single `.py` file - do not create new files
- Maintain logical order and clear structure for readability

## Edge Cases & Error Handling

### State Management
- Track variables and imports carefully to avoid conflicts
- Clear or reinitialize variables before redefining if needed
- Avoid global state pollution

### Resource Management
- Set timeouts for long-running operations (use `signal` or timeout decorators)
- Implement memory checks for large datasets (`psutil.virtual_memory()`)
- Close file handles, database connections, and network resources explicitly
- Use context managers (`with` statements) when possible

### Import & Dependency Issues
- Check if packages are installed before importing (`importlib.util.find_spec()`)
- Provide installation commands if packages are missing
- Handle version conflicts gracefully
- Use try/except blocks around imports with fallback options

### Output Management
- Use logging instead of print statements for better control
- Truncate excessively long outputs in logs
- Save large plots or data outputs to files instead of printing

### File System Operations
- Always check if files/directories exist before operations
- Use absolute paths or validate relative paths
- Handle permission errors gracefully
- Implement file locking for concurrent access scenarios

### Data Handling
- Validate data types and shapes before processing
- Handle missing/NaN values explicitly
- Check for empty datasets before analysis
- Implement data size limits to prevent memory issues

### Interactive Elements
- Avoid code requiring user input (`input()`)
- Manage plot backends consistently
- Clear previous plots when creating new ones (`plt.clf()`)

## Error Recovery Patterns

### Robust Code Block Template

```python
try:
    # Main code logic
    pass
except SpecificError as e:
    # Handle specific known errors
    logging.error(f"Expected error: {e}")
    # Fallback logic
except Exception as e:
    # Handle unexpected errors
    logging.error(f"Unexpected error: {e}")
    # Cleanup logic
    # Alternative approach
finally:
    # Cleanup resources
    pass
```

## Additional Safeguards

- Validate inputs before processing
- Set reasonable default values
- Implement progress tracking for long operations
- Use logging instead of print statements for debugging
- Test code with edge case data before full execution
- Implement rollback mechanisms for destructive operations

## Best Practices Summary

### DO:
✅ Override code blocks/functions when fixing errors  
✅ Include detailed comments/documentation  
✅ Execute code immediately after generation/modification  
✅ Handle errors gracefully with try/except  
✅ Clean up resources properly  
✅ Validate data before processing  

### DON'T:
❌ Create new files  
❌ Append error fixes below original code  
❌ Leave resources unclosed  
❌ Ignore error handling  
❌ Use blocking user input functions  
❌ Create excessively long outputs  

---

*Generated for Python Script Agent Custom Instructions*
