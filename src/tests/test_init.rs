//! #117-5: the `init()` default log filter must match its documented
//! contract ("default to info"), so a library consumer does not get
//! chatty Debug output unless `RUST_LOG` opts in.

#[test]
fn init_defaults_to_info_filter_not_debug() {
    if std::env::var_os("RUST_LOG").is_some() {
        // RUST_LOG overrides the default; the assertion below only holds
        // for the documented default path.
        return;
    }

    crate::init();

    // env_logger derives the global max level from the default filter.
    // `info` keeps Info visible and Debug silent.
    assert!(
        log::log_enabled!(log::Level::Info),
        "Info must be enabled by the default filter"
    );
    assert!(
        !log::log_enabled!(log::Level::Debug),
        "Debug must be silent under the default `info` filter"
    );
}
