# 🚀 Software Engineering Checklist for Data Professionals

## 📋 About This Checklist

Whether you're transitioning from data science to ML engineering, coming from a non-CS background, or just want to write code that doesn't make senior engineers cry during code review 😅, this checklist will guide you through essential software engineering practices.

**Who is this for?**

- 🎓 Entry-level Python developers and data scientists
- 📊 Data professionals who want to ship production code

---

## 🟢 Beginner Level

### Project Setup & Organization

- [ ] **Use a consistent project structure** - No more `untitled_notebook_final_v2_FINAL.ipynb`

  - Separate source code (`src/`), tests (`tests/`), configs (`config/`), and notebooks (`notebooks/`)
  - Check out [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for templates

- [ ] **Create a proper README.md** - Future you will thank present you

  - Include project description, setup instructions, and how to run the code
  - Bonus points: Add badges for build status, coverage, and Python version

- [ ] **Use a `.gitignore` file** - Stop committing your `.DS_Store` and `__pycache__` folders

  - Use [gitignore.io](https://gitignore.io) to generate one for Python/Data Science projects
  - Add your `.env` files, data folders, and model artifacts

- [ ] **Pin your dependencies** - "Works on my machine" usually means unpinned dependencies
  - Use `requirements.txt` or `pyproject.toml` with **specific versions**
  - Better yet: Use `pip freeze > requirements.txt` after getting things working

### Version Control

- [ ] **Commit early, commit often** - Your git history should tell a story, not a novel

  - Make small, focused commits that do one thing
  - Write descriptive commit messages (not "fixed stuff" or "updates")

- [ ] **Use meaningful branch names** - `feature/add-model-monitoring` > `test-branch-2`

  - Convention: `feature/`, `bugfix/`, `hotfix/`, `refactor/`
  - Keep branch names lowercase with hyphens

- [ ] **Never commit secrets** - Hardcoded API keys are a security breach waiting to happen

  - Use environment variables or tools like `python-dotenv`
  - Check files with `git diff` before committing

- [ ] **Write meaningful commit messages** - Follow the 50/72 rule

  ```
  Short summary (50 chars or less)

  More detailed explanation if needed (wrap at 72 chars).
  Explain what and why, not how.
  ```

### Code Quality Basics

- [ ] **Follow PEP 8** - Python has style guidelines for a reason

  - Use a linter like `pylint`, `flake8`, or `ruff` to catch style issues
  - Configure your IDE to format on save with `black` or `ruff format`

- [ ] **Write descriptive variable names** - `df2` and `temp` are not your friends

  - `customer_churn_predictions` > `predictions`
  - `model_accuracy_threshold` > `threshold`

- [ ] **Add docstrings to functions** - Especially if they'll live beyond a notebook

  ```python
  def calculate_rmse(y_true, y_pred):
      """
      Calculate Root Mean Squared Error.

      Args:
          y_true: Array of actual values
          y_pred: Array of predicted values

      Returns:
          float: RMSE value
      """
  ```

- [ ] **Break up long functions** - If you need to scroll, it's too long

  - One function = one responsibility
  - Aim for functions under 20-30 lines

- [ ] **Use type hints** - Help your IDE help you
  ```python
  def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
      ...
  ```

### Testing Fundamentals

- [ ] **Write at least one test** - The hardest test to write is the first one

  - Start with `pytest` and test your most critical function
  - A test is better than no test

- [ ] **Test edge cases** - Empty lists, None values, negative numbers

  - What happens when your input is empty?
  - What if someone passes a string instead of an int?

- [ ] **Use assertions in notebooks** - Sanity checks FTW
  ```python
  assert df.shape[0] > 0, "DataFrame is empty!"
  assert not df['user_id'].duplicated().any(), "Duplicate user_ids found"
  ```

### Environment Management

- [ ] **Use virtual environments** - Global package chaos is real

  - `venv`, `conda`, or `poetry` - just pick one and stick with it
  - Create a new environment for each project

- [ ] **Document your Python version** - Python 3.9 code might not work in 3.12
  - Add it to your README and `.python-version` file
  - Use tools like `pyenv` to manage multiple Python versions

### Configuration Management

- [ ] **Externalize configuration** - No more hardcoded paths and parameters

  - Use config files (`.yaml`, `.json`, `.toml`) or environment variables
  - Make it easy to switch between dev/staging/prod settings

- [ ] **Use pathlib instead of string paths** - It's 2025, act like it 😎
  ```python
  from pathlib import Path
  data_path = Path("data") / "raw" / "customers.csv"
  ```

---

## 🟡 Intermediate Level

### Advanced Version Control

- [ ] **Use feature branches and PRs** - Main branch is sacred

  - Never commit directly to `main` (protect it in GitHub settings)
  - Use pull requests for code review, even if you're solo

- [ ] **Write good PR descriptions** - Context is everything

  - What problem does this solve?
  - What's the approach?
  - How did you test it?

- [ ] **Keep commits atomic** - One logical change per commit

  - Use `git add -p` to stage specific changes
  - Rebase and squash when needed (but don't rewrite public history!)

- [ ] **Use git hooks** - Automate the boring stuff
  - Pre-commit hooks for linting and formatting
  - Pre-push hooks for running tests

### Code Architecture & Design

- [ ] **Separate concerns** - Business logic ≠ data access ≠ API layer

  - Don't mix data loading, transformation, and model training in one function
  - Create separate modules: `data.py`, `models.py`, `utils.py`, `training.py`

- [ ] **Use classes when appropriate** - But don't over-engineer

  - Good for: Models, pipelines, data processors with state
  - Overkill for: Simple utility functions

- [ ] **Implement the DRY principle** - Don't Repeat Yourself

  - If you copy-paste code more than twice, make it a function
  - Extract common logic into utilities

- [ ] **Write modular, reusable code** - Your future projects will thank you

  - Functions should be pure when possible (same input = same output)
  - Avoid global state and side effects

- [ ] **Handle errors gracefully** - Try-except is your friend
  ```python
  try:
      model = load_model(model_path)
  except FileNotFoundError:
      logger.error(f"Model not found at {model_path}")
      raise
  ```

### Testing & Quality

- [ ] **Aim for >70% test coverage** - Use `pytest-cov` to measure

  - Test critical paths first (data loading, model training, inference)
  - Don't obsess over 100% coverage

- [ ] **Write integration tests** - Unit tests aren't enough

  - Test the full pipeline: data loading → preprocessing → training → prediction
  - Use sample data or fixtures

- [ ] **Use fixtures and mocks** - Don't hit real databases in tests

  - Mock external API calls and database connections
  - Create reusable test data with `pytest` fixtures

- [ ] **Set up CI/CD for tests** - If it's not automated, it won't happen
  - GitHub Actions, GitLab CI, or CircleCI
  - Run tests on every PR

### Logging & Monitoring

- [ ] **Use proper logging** - Print statements are for debugging only

  ```python
  import logging
  logging.info(f"Training model with {len(X_train)} samples")
  logging.warning(f"Missing values found: {df.isna().sum().sum()}")
  ```

- [ ] **Log at appropriate levels** - DEBUG, INFO, WARNING, ERROR, CRITICAL

  - Use DEBUG for development, INFO for production
  - ERROR for things that break, WARNING for things that might break

- [ ] **Log, don't print** - Logs can be filtered, aggregated, and monitored
  - Configure logging to write to files in production
  - Include timestamps and log levels

### Performance & Optimization

- [ ] **Profile before optimizing** - "Premature optimization is the root of all evil"

  - Use `cProfile`, `line_profiler`, or `memory_profiler`
  - Focus on the bottlenecks, not guesses

- [ ] **Write efficient pandas code** - Vectorize, don't iterate

  - Use `.apply()` sparingly (it's often a for loop in disguise)
  - Prefer vectorized operations and `.query()` for filtering

- [ ] **Consider memory usage** - Your laptop has limits
  - Use `df.memory_usage()` to check DataFrame size
  - Load data in chunks for large files
  - Use appropriate dtypes (`category` for strings with few unique values)

### Dependencies & Packaging

- [ ] **Use a dependency manager** - `pip-tools`, `poetry`, or `conda`

  - Separate dev dependencies from production
  - Lock your dependencies for reproducibility

- [ ] **Make your code pip-installable** - Create a proper package

  - Add a `setup.py` or `pyproject.toml`
  - Makes imports cleaner: `from my_project import utils`

- [ ] **Document your environment setup** - Include OS-specific gotchas
  - Different instructions for macOS, Linux, Windows if needed
  - Document any system dependencies (e.g., `libhdf5`)

### Code Review & Collaboration

- [ ] **Review your own code first** - Be your own harsh critic

  - Read through the diff before requesting review
  - Look for: TODO comments, debug code, commented-out code

- [ ] **Leave helpful PR comments** - Be specific and constructive

  - "Consider using list comprehension here for better performance"
  - Not "this code is bad"

- [ ] **Accept feedback graciously** - Code review isn't personal
  - "Good catch!" > "but it works though..."
  - Every review is a learning opportunity

### Production Readiness

- [ ] **Add health checks** - Know when your service is down

  - Simple endpoint that returns 200 if everything's working
  - Check database connections, model loading, etc.

- [ ] **Implement graceful degradation** - Fail softly when possible

  - Return cached predictions if model is unavailable
  - Provide meaningful error messages

- [ ] **Use containerization** - Docker is your friend

  - Makes "works on my machine" problems disappear
  - Pin base image versions

- [ ] **Implement retry logic** - Networks are unreliable
  - Retry failed API calls with exponential backoff
  - Use libraries like `tenacity` or `backoff`

---

## 🎯 Quick Wins for Immediate Impact

If you're overwhelmed, start here:

1. ✅ Set up a linter and auto-formatter (black + ruff)
2. ✅ Write a proper README with setup instructions
3. ✅ Add type hints to function signatures
4. ✅ Write one test for your most critical function
5. ✅ Use environment variables for secrets and configs
6. ✅ Add logging instead of print statements
7. ✅ Create a `.gitignore` file
8. ✅ Set up pre-commit hooks

---

## 📚 Recommended Resources

- **Books**: "Clean Code" (Robert Martin), "The Pragmatic Programmer" (Hunt & Thomas)
- **Blogs**: [Real Python](https://realpython.com), [cynscode.com](https://cynscode.com) 😉
- **Tools**: [pre-commit](https://pre-commit.com), [ruff](https://docs.astral.sh/ruff/), [pytest](https://pytest.org)
- **Courses**: Checkout the content on [cynscode.com](https://cynscode.com) for practical guides

---

## 📬 Stay Connected

Want more practical software engineering and ML tips?

- 📝 Read the blog: [cynscode.com](https://cynscode.com)
- 💼 Follow on LinkedIn: [Cynthia Ukawu](https://www.linkedin.com/in/cynthiiaa/)

---
