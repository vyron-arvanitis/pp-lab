# Tips for Using Git in the Course
---

## Cloning the repository
```
git clone <this-repository-url>
```

You can find the url by clicking on the Clone button on the gitlab repository:

![](figures/git_clone_url2.png)

## Add ssh keys

If you don't want to enter your password every time you interact with the git repository you can use ssh keys - there is a [detailed documentation on a gitlab help page](https://gitlab.physik.uni-muenchen.de/help/user/ssh.md). Quickstart for the setup at LMU:

```sh
ssh-keygen -t ed25519 #generate a new key pair (only need to do once)
cat ~/.ssh/id_ed25519.pub #print public key, to add to profile go to (Top right corner -> Edit profile -> Left bar -> SSH keys)
ssh-add #add key to ssh-agent (sort of a keyring - only to be done once per session)
```

## Creating a new branch
```sh
cd <code-directory>   #change to cloned directory 
git branch your_individual_branch_name   #create branch
git checkout your_individual_branch_name   #switch to branch
git push -u origin your_individual_branch_name   #creates and tracks branch on remote (git will also suggest that command if you only type `git push` the first time you push)
```

## Making Changes
```sh
git status   #optional: check which files you've changed
git add <path>   #list files you want to stage for commit
git status #optional: check which files you have staged
git commit -m "commit message"
git push #push to remote
```

## Checking out other people's work
You can "copy" a file from another branch using
```sh
git checkout <branch-name> -- <file-name>
```
But maybe you have a file with the same name (e.g. when checking out somebody else's `workbook.ipynb`). Then you can use the following command:
```sh
git show <branch-name>:<file-name> > <new-file-name>
```
(notice the **>** redirection operator)  

If somebody else has created a new branch that you want to check out, do this:
```sh
git fetch   #retrieves all changes from the remote
git branch -a   #view all existing branches (remote & local)
git checkout <branch-name>
```

## Temporarily move changes away
You may want to checkout a different branch but have local changes you didn't (and don't want to) commit just yet. Git will give you a message like
```
error: Your local changes to the following files would be overwritten by checkout:
        ...
Please commit your changes or stash them before you switch branches.
Aborting
```
One way to handle this is to use `git stash`, e.g.

```sh
git stash #temporarily move changes away
git checkout <some-other-branch>
# ... do work there
git checkout <original-branch>
git stash pop #get temporary changes back
```


## Merging
When you are working with others and you want to implement something new, it makes sense to first create another branch and try those changes there. Once you are done, don't try and merge these changes to the main branch (or whatever branch everybody is working on) directly. Merge the main branch into yours first:  
`git merge <branch-to-merge-from> --no-edit`  
and resolve any conflicts. Then go to the GitLab page and create a Merge Request (merging your branch into main).

## Solving Merge Conflicts
Conflicts get written to the file like this:  
```
<<<<<<< HEAD
<code.version(1)>
=======
<code.version(2)>
>>>>>>> other-branch
```
Remove the markings and the version of the code you don't want to keep. Save the file and:
```sh
git add <filename>   #stage changes in file with resolved conflicts
git commit --no-edit  #finish the merge, no commit message
# or abort the merge:
git merge --abort
```

## Ignoring Files
The repo features a preconfigured `.gitignore`, such that git ignores certain file types. For example you DON'T want to add the datasets, because they are way too large. Hence the file type `*.parquet` is specified in gitignore.  
You can also add single files: `<filename>`  
And entire subdirectories: `<directory-name>/`  
To exclude a certain file from being ignored: `!<filename>`  
If you want to ignore a file that is already being tracked, add it to `.gitignore` and do `git rm --cached <file>`.

## Amending
add (small!) changes to last commit without making a new one:   
`git add <file>`    
`git commit --amend --no-edit`


## Git "Rules"
* **Meaningful names and commit messages**:  
Just like with objects in Python, name your files and branches after their respective purpose. For commit messages, describe what each commit does, but keep it short and concise (easier for smaller commits, which you should do anyway).
* **No large files**:  
Even if they are only on your branch, everybody updating their repository will have to download them and will be very annoyed. It is better to provide links so that others can get these files only if they need to. To have git ignore certain files, see above section "Ignoring Files". To be clear, "large" means more than ~10MB in this case. Be careful with many medium sized files like plots: they can add up to a significant size as well. If they can be generated from the existing code, ignore them as well.
* **Check Status**:  
Frequently do `git status` to check if your branch is up to date, everything is staged that you want to be staged, etc.
* **Small and separate commits**:  
`Commit often, commit early!` Git can only develop it's full potential, if you have many safe states to compare with and go back to. You should also keep your commits logically seperated, so that they only change one "aspect" of your code at a time. Also when working together, you will avert many conflicts and errors if you `push` your commits often, so that others can keep their repos up to date.
* **Pull often**:  
When working together on a branch, you should at least do `git pull` once before editing any code, in order to not change lines that are different on the remote and to avoid subsequent merge conflicts. If a push fails because somebody has pushed something on the remote while you were working, you need to apply these remote changes (and solve potential conflicts) with `git pull --rebase`. (Avoid `pull --merge`, because this unnecessarily creates a second commit.) 

