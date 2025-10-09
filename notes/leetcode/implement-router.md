---
title: "Implement Router"
date: "2025-01-27"
tags: ["Hash Table", "String", "Design"]
related: []
slug: "implement-router"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/implement-router/description/"
leetcode_difficulty: "Medium"
leetcode_topics: ["Hash Table", "String", "Design"]
---

# Implement Router

## Summary
Design and implement a basic HTTP router that can register routes with different HTTP methods and handle incoming requests by matching them to the appropriate handler functions.

## Problem Description
Implement a Router class that supports:

1. **Route Registration**: Register routes with specific HTTP methods (GET, POST, PUT, DELETE, etc.)
2. **Route Matching**: Match incoming requests to registered routes
3. **Handler Execution**: Execute the appropriate handler function for matched routes
4. **404 Handling**: Return appropriate response for unmatched routes

**Key Requirements:**
- Support multiple HTTP methods
- Handle exact path matching
- Support path parameters (e.g., `/users/:id`)
- Return appropriate HTTP status codes
- Handle edge cases like duplicate routes, invalid methods

**Example Usage:**
```python
router = Router()

# Register routes
router.add_route('GET', '/users', get_users_handler)
router.add_route('GET', '/users/:id', get_user_handler)
router.add_route('POST', '/users', create_user_handler)

# Handle requests
response = router.handle_request('GET', '/users')
response = router.handle_request('GET', '/users/123')
response = router.handle_request('POST', '/users')
```

## Solution Approach
This problem can be solved using a hash table-based approach with support for path parameters:

### Core Data Structure
- Use a dictionary where keys are HTTP methods and values are dictionaries of route patterns to handlers
- Store both exact matches and parameterized routes
- Use regex patterns for parameter matching

### Algorithm Steps
1. **Route Registration**:
   - Parse route pattern to identify parameters (e.g., `:id`)
   - Convert to regex pattern for matching
   - Store handler function with compiled regex

2. **Route Matching**:
   - Look up routes for the given HTTP method
   - Try exact matches first
   - Try parameterized routes using regex matching
   - Extract parameter values from matched route

3. **Handler Execution**:
   - Execute the matched handler with extracted parameters
   - Return appropriate response object

## Time & Space Complexity
- **Time Complexity:**
  - Route registration: O(1) average case
  - Route matching: O(n) where n is number of registered routes for the method
  - Handler execution: O(1)
- **Space Complexity:** O(m × r) where m is number of HTTP methods and r is number of routes per method

## Key Insights
- **Exact match priority**: Check exact matches before parameterized routes
- **Parameter extraction**: Use regex groups to extract parameter values
- **Method separation**: Organize routes by HTTP method for efficient lookup
- **Error handling**: Proper 404 and 405 (Method Not Allowed) responses
- **Route conflicts**: Handle duplicate route registrations appropriately

## Examples / snippets

### Solution Code
```python
import re
from typing import Dict, Callable, Optional, Tuple, Any
from collections import defaultdict

class Router:
    def __init__(self):
        # Dictionary: method -> {pattern: (handler, compiled_regex, param_names)}
        self.routes = defaultdict(dict)
        # Dictionary: method -> {exact_path: handler} for exact matches
        self.exact_routes = defaultdict(dict)
    
    def add_route(self, method: str, pattern: str, handler: Callable) -> None:
        """Register a route with the given method, pattern, and handler."""
        method = method.upper()
        
        # Check for exact match (no parameters)
        if ':' not in pattern:
            if pattern in self.exact_routes[method]:
                raise ValueError(f"Route {method} {pattern} already exists")
            self.exact_routes[method][pattern] = handler
            return
        
        # Handle parameterized routes
        if pattern in self.routes[method]:
            raise ValueError(f"Route {method} {pattern} already exists")
        
        # Convert pattern to regex and extract parameter names
        param_names = []
        regex_pattern = pattern
        
        # Find all :param patterns
        param_matches = re.findall(r':(\w+)', pattern)
        param_names = param_matches
        
        # Replace :param with regex groups
        regex_pattern = re.sub(r':(\w+)', r'([^/]+)', pattern)
        regex_pattern = f'^{regex_pattern}$'
        
        compiled_regex = re.compile(regex_pattern)
        self.routes[method][pattern] = (handler, compiled_regex, param_names)
    
    def handle_request(self, method: str, path: str) -> Dict[str, Any]:
        """Handle an incoming request and return the response."""
        method = method.upper()
        
        # Try exact match first
        if path in self.exact_routes[method]:
            handler = self.exact_routes[method][path]
            return self._execute_handler(handler, {})
        
        # Try parameterized routes
        for pattern, (handler, compiled_regex, param_names) in self.routes[method].items():
            match = compiled_regex.match(path)
            if match:
                # Extract parameter values
                params = {}
                for i, param_name in enumerate(param_names):
                    params[param_name] = match.group(i + 1)
                
                return self._execute_handler(handler, params)
        
        # Check if method exists but path doesn't (405 Method Not Allowed)
        if self.exact_routes[method] or self.routes[method]:
            return {
                'status_code': 405,
                'message': 'Method Not Allowed',
                'data': None
            }
        
        # Route not found (404 Not Found)
        return {
            'status_code': 404,
            'message': 'Not Found',
            'data': None
        }
    
    def _execute_handler(self, handler: Callable, params: Dict[str, str]) -> Dict[str, Any]:
        """Execute the handler function with extracted parameters."""
        try:
            if params:
                result = handler(**params)
            else:
                result = handler()
            
            return {
                'status_code': 200,
                'message': 'OK',
                'data': result
            }
        except Exception as e:
            return {
                'status_code': 500,
                'message': 'Internal Server Error',
                'data': str(e)
            }
    
    def list_routes(self) -> Dict[str, list]:
        """List all registered routes for debugging."""
        all_routes = {}
        for method in self.exact_routes:
            all_routes[method] = list(self.exact_routes[method].keys())
        for method in self.routes:
            if method not in all_routes:
                all_routes[method] = []
            all_routes[method].extend(self.routes[method].keys())
        return all_routes

# Example usage and test handlers
def get_users():
    return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

def get_user(id: str):
    return {"user": {"id": int(id), "name": f"User {id}"}}

def create_user():
    return {"message": "User created successfully"}

def update_user(id: str):
    return {"message": f"User {id} updated successfully"}

def delete_user(id: str):
    return {"message": f"User {id} deleted successfully"}

# Example usage
if __name__ == "__main__":
    router = Router()
    
    # Register routes
    router.add_route('GET', '/users', get_users)
    router.add_route('GET', '/users/:id', get_user)
    router.add_route('POST', '/users', create_user)
    router.add_route('PUT', '/users/:id', update_user)
    router.add_route('DELETE', '/users/:id', delete_user)
    
    # Test requests
    print(router.handle_request('GET', '/users'))
    print(router.handle_request('GET', '/users/123'))
    print(router.handle_request('POST', '/users'))
    print(router.handle_request('PUT', '/users/456'))
    print(router.handle_request('DELETE', '/users/789'))
    print(router.handle_request('GET', '/nonexistent'))
    print(router.handle_request('PATCH', '/users'))
```

### Example Walkthrough
**Example 1: Basic Route Registration and Matching**
```python
router = Router()

# Register routes
router.add_route('GET', '/users', get_users_handler)
router.add_route('GET', '/users/:id', get_user_handler)

# Handle requests
response1 = router.handle_request('GET', '/users')
# Returns: {'status_code': 200, 'message': 'OK', 'data': {...}}

response2 = router.handle_request('GET', '/users/123')
# Returns: {'status_code': 200, 'message': 'OK', 'data': {'user': {'id': 123, 'name': 'User 123'}}}
```

**Example 2: Error Handling**
```python
# 404 Not Found
response = router.handle_request('GET', '/nonexistent')
# Returns: {'status_code': 404, 'message': 'Not Found', 'data': None}

# 405 Method Not Allowed
response = router.handle_request('PATCH', '/users')
# Returns: {'status_code': 405, 'message': 'Method Not Allowed', 'data': None}
```

**Example 3: Multiple Parameters**
```python
def get_user_posts(user_id: str, post_id: str):
    return {"user_id": user_id, "post_id": post_id, "content": "Post content"}

router.add_route('GET', '/users/:user_id/posts/:post_id', get_user_posts)

response = router.handle_request('GET', '/users/123/posts/456')
# Returns: {'status_code': 200, 'message': 'OK', 'data': {'user_id': '123', 'post_id': '456', 'content': 'Post content'}}
```

## Edge Cases & Validation
- **Duplicate routes**: Should raise ValueError when registering duplicate routes
- **Invalid HTTP methods**: Should handle case-insensitive method matching
- **Empty paths**: Should handle root path `/` appropriately
- **Special characters**: Should handle paths with special characters correctly
- **Parameter validation**: Should extract parameters correctly from complex paths
- **Handler exceptions**: Should catch and return 500 errors for handler exceptions
- **Method not allowed**: Should return 405 when method exists but path doesn't
- **Case sensitivity**: Should handle path matching case-sensitively
- **Trailing slashes**: Should handle paths with and without trailing slashes consistently

## Related Problems
- [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/) - Tree-based routing
- [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/) - Pattern matching
- [676. Implement Magic Dictionary](https://leetcode.com/problems/implement-magic-dictionary/) - String matching with variations
- [Design Twitter](https://leetcode.com/problems/design-twitter/) - System design with routing concepts
