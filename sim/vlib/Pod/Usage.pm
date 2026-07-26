# Minimal stub for Pod::Usage — the oss-cad MSYS perl 5.38 ships without the
# Pod/ core modules. The verilator wrapper `use Pod::Usage;` unconditionally,
# but only *calls* pod2usage on --help / arg-parse errors. We never invoke
# --help and pass valid args, so a stub that prints the message and exits is
# enough to let the verilator Perl wrapper run.
package Pod::Usage;
use strict;
use warnings;
sub pod2usage {
    my %a = @_;
    my $msg = $a{-message};
    my $exit = defined $a{-exitstatus} ? $a{-exitstatus} : (defined $a{-exitval} ? $a{-exitval} : 2);
    if (defined $msg) {
        print STDERR $msg, "\n";
    } else {
        print STDERR "pod2usage: see verilator --help (stub)\n";
    }
    exit($exit);
}
1;