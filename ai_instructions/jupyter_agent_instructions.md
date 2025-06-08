# Custom Instructions for Jupyter Notebook Agent

## Core Instructions

### Code Generation & Documentation

**Before Each Code Cell:**
- Create a detailed markdown cell explaining:
  - What the code will accomplish
  - Key functions/methods being used
  - Expected inputs/outputs
  - Any assumptions or prerequisites

**After Execution:**
- Add a markdown cell analyzing:
  - What the results mean
  - Any unexpected outcomes
  - Next steps or implications

### Execution Protocol

- Execute code immediately after generation
- **REPLACE/OVERRIDE** existing cells when fixing errors - never append below
- Work within the current notebook - do not create new notebooks
- Maintain execution order and cell numbering consistency

## Edge Cases & Error Handling

### State Management
- Track variables and imports across cells to avoid conflicts
- Clear problematic variables before redefining
- Use `%reset -f` sparingly and only when necessary
- Check for existing variable names before assignment

### Resource Management
- Set timeouts for long-running operations (use `signal` or `timeout` decorators)
- Implement memory checks for large datasets (`psutil.virtual_memory()`)
- Close file handles, database connections, and network resources explicitly
- Use context managers (`with` statements) when possible

### Import & Dependency Issues
- Check if packages are installed before importing (`importlib.util.find_spec()`)
- Provide installation commands if packages are missing
- Handle version conflicts gracefully
- Use try/except blocks around imports with fallback options

### Output Management
- Truncate excessively long outputs (set `pd.options.display.max_rows = 50`)
- Handle large plots by saving to files instead of inline display
- Suppress verbose outputs when appropriate (`%%capture` magic)
- Use `display()` for multiple outputs in single cells

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
- Handle widget state persistence
- Manage plot backends (matplotlib, plotly) consistently
- Clear previous plots when creating new ones (`plt.clf()`)

## Error Recovery Patterns

### Robust Cell Structure Template

```python
try:
    # Main code logic
    pass
except SpecificError as e:
    # Handle specific known errors
    print(f"Expected error: {e}")
    # Fallback logic
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
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
✅ Override cells when fixing errors  
✅ Include detailed markdown documentation  
✅ Execute code immediately after generation  
✅ Handle errors gracefully with try/except  
✅ Clean up resources properly  
✅ Validate data before processing  

### DON'T:
❌ Create new notebooks  
❌ Append error fixes below original code  
❌ Leave resources unclosed  
❌ Ignore error handling  
❌ Use blocking user input functions  
❌ Create excessively long outputs  

---

*Generated for Jupyter Notebook Agent Custom Instructions*
