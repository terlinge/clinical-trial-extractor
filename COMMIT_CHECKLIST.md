# 📋 Commit Checklist - Clinical Trial Extractor

**Always run through this checklist before making any commit to ensure documentation stays current.**

---

## 🔄 **Pre-Commit Documentation Updates**

### 1. **Code Changes Documentation**
- [ ] Update `CHANGELOG.md` with new features/fixes
- [ ] Update version numbers in relevant files
- [ ] Update `README.md` if features/usage changed
- [ ] Update `requirements.txt` if dependencies changed
- [ ] Update `INSTALLATION.md` if setup process changed

### 2. **Database Schema Changes**
- [ ] Update database schema documentation in `README.md`
- [ ] Update `database_query_examples.py` with new query patterns
- [ ] Document any migration steps needed
- [ ] Update field descriptions and relationships

### 3. **API Changes**
- [ ] Update API endpoint documentation
- [ ] Update example requests/responses
- [ ] Document any breaking changes
- [ ] Update error handling documentation

### 4. **Configuration Changes**
- [ ] Update environment variable documentation
- [ ] Update configuration examples
- [ ] Document new settings or changed defaults
- [ ] Update deployment instructions if needed

---

## 📝 **Commit Message Standards**

### **Format Template:**
```
🎯 [TYPE] Brief Description

📊 CHANGES:
- Bullet point of main changes
- Another key change
- Database/API modifications

🔧 TECHNICAL:
- Technical implementation details
- Performance improvements
- Bug fixes

📚 DOCUMENTATION:
- README updates
- New guides added
- Configuration changes

✅ TESTING:
- Tests added/updated
- Validation steps
- Quality assurance notes
```

### **Commit Types:**
- `🚀 FEATURE:` New functionality
- `🐛 BUGFIX:` Bug fixes and corrections
- `📚 DOCS:` Documentation updates only
- `🔧 REFACTOR:` Code improvements without new features
- `⚡ PERFORMANCE:` Performance optimizations
- `🔐 SECURITY:` Security improvements
- `🗄️ DATABASE:` Schema changes or migrations
- `🎨 UI:` User interface improvements
- `📦 DEPENDENCIES:` Package updates

---

## 📋 **Documentation Update Checklist**

### **When Adding New Features:**
- [ ] Add feature description to `README.md`
- [ ] Update feature list in `README.md`
- [ ] Add usage examples if applicable
- [ ] Update `CHANGELOG.md` with feature details
- [ ] Update `requirements.txt` if new dependencies
- [ ] Add configuration documentation if needed

### **When Fixing Bugs:**
- [ ] Update `CHANGELOG.md` with bug fix details
- [ ] Update troubleshooting section in documentation
- [ ] Document any workarounds that are no longer needed
- [ ] Update known issues list

### **When Changing Database Schema:**
- [ ] Update database schema diagram in `README.md`
- [ ] Update field descriptions and relationships
- [ ] Add new query examples to `database_query_examples.py`
- [ ] Document migration steps in `INSTALLATION.md`
- [ ] Update backup/restore procedures if affected

### **When Updating Dependencies:**
- [ ] Update `requirements.txt` with version changes
- [ ] Update installation instructions if needed
- [ ] Update troubleshooting for dependency issues
- [ ] Document any breaking changes from dependency updates

### **When Changing Configuration:**
- [ ] Update environment variable documentation
- [ ] Update `START_APP.ps1` or startup scripts if needed
- [ ] Update configuration examples
- [ ] Document any new required settings

---

## 🎯 **Specific File Update Guidelines**

### **README.md Updates Needed When:**
- Adding new features or capabilities
- Changing system requirements
- Modifying installation process
- Updating usage instructions
- Changing database schema
- Modifying API endpoints

### **CHANGELOG.md Updates Needed When:**
- Every commit (document all changes)
- Version releases (add version header)
- Breaking changes (mark clearly)
- Performance improvements
- Bug fixes with impact

### **requirements.txt Updates Needed When:**
- Adding new Python packages
- Updating package versions
- Removing unused dependencies
- Adding optional dependencies
- Platform-specific requirements change

### **INSTALLATION.md Updates Needed When:**
- Changing system requirements
- Modifying setup steps
- Adding new configuration steps
- Updating database setup
- Changing environment variables

### **database_query_examples.py Updates Needed When:**
- Adding new database tables
- Adding new query capabilities
- Creating new indexes
- Changing query patterns
- Adding complex query examples

---

## 🚀 **Release Preparation Checklist**

### **Major Version Release (x.0.0):**
- [ ] Complete `CHANGELOG.md` with all changes since last major version
- [ ] Update version numbers in all relevant files
- [ ] Update `README.md` with comprehensive feature overview
- [ ] Verify all documentation is current and accurate
- [ ] Update installation and setup guides
- [ ] Test all documented examples and code samples
- [ ] Update dependency versions and compatibility notes

### **Minor Version Release (x.y.0):**
- [ ] Update `CHANGELOG.md` with new features
- [ ] Update `README.md` feature list
- [ ] Update usage examples if changed
- [ ] Update configuration documentation if needed
- [ ] Test installation process

### **Patch Release (x.y.z):**
- [ ] Update `CHANGELOG.md` with bug fixes
- [ ] Update troubleshooting documentation if applicable
- [ ] Verify fix doesn't break documented examples

---

## 🔍 **Pre-Commit Validation Steps**

### **Documentation Consistency Check:**
1. [ ] Run through README instructions to verify they work
2. [ ] Check all code examples in documentation still work
3. [ ] Verify all links in documentation are functional
4. [ ] Ensure version numbers are consistent across files
5. [ ] Check that CHANGELOG dates and versions are correct

### **Code Quality Check:**
1. [ ] Ensure all new code has appropriate comments
2. [ ] Verify error handling is documented
3. [ ] Check that new endpoints/functions have docstrings
4. [ ] Ensure configuration options are documented

### **Dependency Check:**
1. [ ] Verify all imports are in requirements.txt
2. [ ] Check for unused dependencies
3. [ ] Ensure version constraints are appropriate
4. [ ] Test installation from clean environment if major changes

---

## 📚 **Documentation Standards**

### **Writing Guidelines:**
- Use clear, concise language
- Include practical examples
- Provide step-by-step instructions
- Use consistent formatting and terminology
- Include error handling and troubleshooting
- Keep technical accuracy as top priority

### **Code Examples:**
- Always test code examples before documenting
- Include expected output where helpful
- Show error handling patterns
- Use realistic sample data
- Include context and prerequisites

### **Version Documentation:**
- Date all version releases
- Categorize changes (Features, Bugfixes, Breaking Changes)
- Include upgrade/migration instructions
- Document any new requirements or dependencies
- Highlight security-related changes

---

## ✅ **Final Commit Command Template**

```bash
# 1. Stage all changes
git add .

# 2. Use comprehensive commit message following template above
git commit -m "🎯 [TYPE] Brief Description

📊 CHANGES:
- List of main changes
- Database modifications
- API updates

📚 DOCUMENTATION:
- README updates
- New documentation files
- Configuration changes

✅ VALIDATION:
- Testing completed
- Documentation verified
- Examples validated"

# 3. Push to repository
git push origin main
```

---

## 🎯 **Remember: Documentation is Code**

Treat documentation updates with the same rigor as code changes:
- Review documentation changes carefully
- Test all documented procedures
- Keep documentation in sync with functionality
- Update examples when APIs change
- Maintain consistency across all documentation files

**The goal is that anyone can understand, install, and use the system based solely on the documentation provided.**