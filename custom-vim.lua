function makeprg_release()
	vim.o.makeprg = 'make release'
	vim.cmd("make")
end

function makeprg_debug()
	vim.o.makeprg = 'make debug'
	vim.cmd("make")
end

function makeprg_release_optimized()
	vim.o.makeprg = 'make release-optimize'
	vim.cmd("make")
end

--vim.keymap.set('i', '<Tab>', '<Tab>', { noremap = true, silent = true })
-- set noexpandtab but in lua
vim.opt_local.expandtab = false
vim.opt_local.shiftwidth = 3
vim.opt_local.tabstop = 3

function clean_file()
	vim.o.makeprg = 'clang-tidy ' .. vim.fn.expand('%')
	local oldEfm = vim.o.errorformat
	vim.o.errorformat = '%f:%l:%c: %t%*[^:]: %m'
	vim.cmd('make')
	vim.o.errorformat = oldEfm
end

vim.keymap.set('n', '<leader>ec', clean_file);

function cppcheck_file()
	vim.o.makeprg = (
		'cppcheck --enable=all --inconclusive --std=c++23 '
		.. ' --suppress=missingIncludeSystem -Israt/include '
		.. ' --suppress=unusedFunction '
		.. ' --suppress=noExplicitConstructor '
		.. ' --suppress=funcArgNamesDifferent '
		.. ' --suppress=functionConst '
		.. ' srat/src/'
	)
	local oldEfm = vim.o.errorformat
	vim.o.errorformat = '%f:%l:%c: %t%*[^:]: %m'
	vim.cmd('make')
	vim.o.errorformat = oldEfm
end

vim.keymap.set('n', '<leader>ep', cppcheck_file);

vim.cmd [[
let g:copilot_workspace_folders = ["~/repo/demo/"]
]]
